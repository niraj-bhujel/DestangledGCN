#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:29:58 2020

@author: dl-asoro
"""

import os
import sys

import csv
import time
import copy
import json
import yaml
import math
import shutil
import pickle
import random
import traceback
import secrets
import getpass
import numpy as np
import os.path as osp
from datetime import datetime
from collections import defaultdict

import torch
print('Torch CUDA Devices:', torch.cuda.device_count())

torch.backends.cudnn.benchmark=False #If you do not need reproducibility, performance might improve if benchmarking is enabled. only applicable to CUDA convolution operations.  
torch.backends.cudnn.deterministic=True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["NCCL_DEBUG"] = "INFO"

from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from argument_parser import parse_args

from data.states import *
from data.dataset import PedestriansDataset, NuScenesDataset, WaymoDataset

from model.models import gnn_model
from model.physics import *
from model.losses import loss_func_param, MultiTaskLoss
from utils.early_stopping import EarlyStopping
from utils.schedulers import *

from utils.misc import *
from utils.metrics import *
from utils.graph_utils import *
from utils.train_utils import *
from utils.config import *
from utils.model_utils import *

from visualization.vis_graph import network_draw
from visualization.vis_traj import plot_path, plot_kde

def run(args, cfg, device, prediction_model=None):

    print('Starting Run: %s'%args.out_path)
    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| trial: %s' % args.trial)
    print('| run: %s' % args.run)
    print('| device: %s'%device)
    print('| local rank: %d'%args.local_rank)
    print('| TimeStamp: %s'%args.datetime)
    print('| Trial: %s'%args.out_path)
    print('-----------------------')

    if cfg.model=='STDec':
        from stdec_pipeline import train_val_epoch, evaluate_model
    else:
        from trajdec_pipeline import train_val_epoch, evaluate_model

    logger = create_logger(args.out_path if not args.debug else None)

    # cfg.input_nodes_std_param = get_standaridzation_params(cfg.node_inputs, cfg.standardization.serialize, cfg.node_types)
    # cfg.output_nodes_std_param = get_standaridzation_params(cfg.node_outputs, cfg.standardization.serialize, cfg.node_types)

    datasets = {'pedestrians':PedestriansDataset, 'waymo': WaymoDataset, 'nuscenes': NuScenesDataset}
    datasets = {phase:datasets[cfg.dataset](cfg.data_dir,
                                            phase,
                                            version=cfg.version,
                                            node_types=cfg.node_types,
                                            interaction_radius=INTERACTION_RADIUS,
                                            obsv_len=cfg.obsv_seq_len,
                                            pred_len=cfg.pred_seq_len,
                                            min_obsv_len = cfg.min_obsv_len,
                                            min_seq_len=cfg.min_seq_len,
                                            min_agents=cfg.min_agents,
                                            skip=cfg.skip if phase=='train' else 0,
                                            aug_prob=cfg.augment_prob if phase=='train' else 0, # augment only train data
                                            dt=cfg.dt,
                                            fov_graph = cfg.fov_graph,
                                            include_lane=cfg.include_lane,
                                            preprocess=args.preprocess,
                                            num_process=args.num_process,
                                            ) for phase in cfg.phases}
    # NOTE!! overfit on single batch, used for debugging
    if args.single_batch:
        for _, dataset in datasets.items():
            # dataset.graphs_list = [dataset.graphs_list[i] for i in np.random.choice(len(dataset), cfg.test_bs) ]
            dataset.graphs_list = [dataset.graphs_list[i] for i in np.random.choice(len(dataset), 10) ]

    db_sampler = None
    if args.distributed:
        db_sampler = DistributedSampler(datasets['train'],
                                       shuffle=True,
                                       rank=args.local_rank,
                                       num_replicas=args.world_size)

    logger.info('Creating dataloaders ...')
    # NOTE: shuffle doesn't work with sampler
    dataloaders = {phase:DataLoader(datasets[phase],
                                    batch_size=cfg.test_bs if phase in ['test'] else cfg.train_bs,
                                    num_workers=cfg.num_workers,
                                    shuffle = (db_sampler is None) if phase=='train' else False,
                                    sampler=db_sampler if phase=='train' else None, # distributed sampler only for train loader
                                    collate_fn=datasets[phase].collate,
                                    pin_memory=True) for phase in cfg.phases}

    if cfg.dataset=='nuscenes':
        datasets['test'] = datasets['val']
        dataloaders['test'] = dataloaders['val']

    if cfg.dataset=='pedestrians':
        datasets['val'] = datasets['test']
        dataloaders['val'] = dataloaders['test']

    for phase in dataloaders.keys():
        logger.info('{} dataset: {} samples, {} batches (batch size {})'.format(phase,
                                                                          len(datasets[phase]),
                                                                          len(dataloaders[phase]),
                                                                          dataloaders[phase].batch_size))

    #create model
    logger.info("Creating model ....")
    model = gnn_model(cfg.model, cfg.serialize)
    # model = model.double()
    model = model.to(device)
    logger.info(model)

    if args.distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        find_unused_parameters=True)

    cfg.num_parameters = model_parameters(model, verbose=0)


    for m_name, module in model.named_children():
        print(m_name, model_attributes(module, verbose=0), '\n')

    mtl = MultiTaskLoss(cfg, device)

    #optimizer
    params = [{'params': model.parameters()}]
    if cfg.learn_loss_weights :
        params.append({'params':mtl.parameters(), 'weight_decay':0})
    
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # #lr scheculer
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.lr_reduce_factor, patience=cfg.lr_patience, verbose=True)
    # lr_scheduler = CyclicLR(optimizer, base_lr=cfg.lr*cfg.target_ratio[0], max_lr=cfg.lr*cfg.target_ratio[1], step_size_up=20, mode='triangular2', cycle_momentum=False)
    # lr_scheduler = OneCycleLR(optimizer, max_lr=cfg.lr*cfg.target_ratio[1], epochs=cfg.epochs, steps_per_epoch=1, pct_start=10/cfg.epochs, div_factor=10)
    # lr_scheduler = ReduceLROnPlateauWithWarmup(optimizer, cfg.lr*cfg.target_ratio[0], cfg.lr, cfg.warmup_epochs, mode='min', factor=cfg.lr_reduce_factor, patience=cfg.lr_patience, verbose=True)
    # kl_scheduler = LinearScheduler(0., 1., epochs=cfg.epochs)
    # kl_scheduler = CyclicScheduler(start_value=5, target_value=0, step_size_up=5, mode='triangular')
    # early_stopping
    early_stopping = EarlyStopping(patience=cfg.early_stop_patience,
                                   metric_name=cfg.early_stop_metric,
                                   ckpt_path=args.out_path,
                                   debug=args.debug,
                                   last_epoch=cfg.initial_epoch)

    writer=None
    if not args.debug:
        writer = SummaryWriter(log_dir=args.summary_dir)
        writer.add_scalar('paramaters', cfg.num_parameters, 0)

    h = defaultdict(list)
    h['0xrun'] = [args.run]
    h['0xrank'] = [args.local_rank]

    if args.resume_training:
        # try:
        model, optimizer, lr_scheduler, h, last_epoch = load_ckpt_simple(model, optimizer, lr_scheduler, args.out_path, ckpt_name='full_model_states')
        cfg.initial_epoch = int(last_epoch)+1
        with open(os.path.join(args.out_path, '/history.json'), 'rb') as f:
            h = json.load(f)

        logger.info('Training resumed from epoch:{}'.format(last_epoch))

    try: # catch keyboard interrupt
        last_eval_epoch = 0
        eval_counter = 0
        for epoch in range(cfg.initial_epoch, cfg.epochs+1, 1):
            start=time.time()
            if args.verbose>0:
                print('\nEpoch {}/{}'.format(epoch, cfg.epochs))

            if args.distributed:
                db_sampler.set_epoch(epoch)

            # cfg.kld_loss_wt = kl_scheduler.step(epoch-1)
            # h['train_kld_wt'] = [cfg.kld_loss_wt]

            h = train_val_epoch(cfg, model, optimizer, dataloaders['train'], device, epoch, phase='train', history=h, prediction_model=prediction_model, multi_tasks_loss=mtl)

            # evaluate for local rank 0 only
            if args.local_rank==0:

                with torch.no_grad():
                    h = train_val_epoch(cfg, model, optimizer, dataloaders['val'], device, epoch, phase='val', history=h, prediction_model=prediction_model, multi_tasks_loss=mtl)

                eval_condition1 = epoch==cfg.initial_epoch
                eval_condition2 = epoch<=20 and epoch%cfg.test_interval==0#
                eval_condition3 = epoch>20 and h['val_loss'][-1]<=min(h['val_loss'][-cfg.test_interval:-1])# if minimum in last 5 epochs

                evaluate = eval_condition1 or eval_condition2 or eval_condition3

                if evaluate:
                    h, raw_data_dict = evaluate_model(cfg, model, dataloaders['test'], device, history=h, prediction_model=prediction_model, return_raw_data=False)

                    early_stopping(h[cfg.early_stop_metric][-1], model, epoch)

                    last_eval_epoch = epoch
                    eval_counter += 1

                    h['min_test_ade'] = [h['test_ade'][np.argmin(h['test_ade'])]]
                    h['min_test_fde'] = [h['test_fde'][np.argmin(h['test_fde'])]]

                    h['min_test_ade_best'] = [h['test_ade_best'][np.argmin(h['test_ade_best'])]]
                    h['min_test_fde_best'] = [h['test_fde_best'][np.argmin(h['test_fde_best'])]]

                    if not args.debug:
                        # save fde_best model
                        if h['test_fde_best'][-1]<=h['min_test_fde_best'][-1]: # only < will not work!
                            torch.save(model.state_dict(), args.out_path + '/min_test_fde_best.pth')
                        # save ade_best model
                        if h['test_ade_best'][-1]<=h['min_test_ade_best'][-1]:
                            torch.save(model.state_dict(), args.out_path + '/min_test_ade_best.pth')
                        #save whole model every test interval at the cost of time and memory, helpful when trainning is interrupted
                        save_ckpt_simple(model, optimizer, lr_scheduler, h, epoch, args.out_path, ckpt_name='full_model_states')
                        with open(args.out_path +'/history.json', 'w') as fp:
                            json.dump(h, fp)

                h['epoch'].append(epoch)
                h['lr'].append(optimizer.param_groups[0]['lr'])
                h['eta'].append((time.time()-start))

                if args.verbose>0:
                    logger.info(', '.join(['{}:{:.8s}'.format(k, str(v[-1])) for k, v in sorted(h.items())]))

                # if nan exit training
                for key, val in h.items():
                    if np.isnan(val[-1]):
                        logging.exception('{} is {}'.format(key, val[-1]))
                        # print(', '.join(['{}:{:.8s}'.format(k, str(v[-1])) for k, v in sorted(h.items())]))
                        raise

                if not args.debug:
                    # write test ade/fde at test interval only to avoid ugly step plot
                    if epoch%cfg.test_interval==0 or epoch==1:

                        for key in ['test_ade', 'test_fde', 'test_ade_best', 'test_fde_best']:
                            if epoch>20 and eval_counter>0:
                                writer.add_scalar(key, min(h[key][-eval_counter:]), epoch)
                            else:
                                writer.add_scalar(key, h[key][-1], epoch)
                        # reset
                        eval_counter = 0
                    
                    # write remaining scalars
                    for key, val in h.items():
                        if key in ['min_test_ade', 'min_test_fde', 'min_test_ade_best', 'min_test_fde_best']:
                            continue
                        
                        if key in ['test_ade', 'test_fde', 'test_ade_best', 'test_fde_best', 'epoch', '0xrun', '0xrank']:
                            continue
                            
                        writer.add_scalar(key, val[-1], epoch)

                    # write mtl weights parameters
                    for s in cfg.node_outputs:
                        writer.add_scalar(f'loss_weights/{s}', mtl.node_loss_wt[s].item(), epoch)
                    for s in cfg.edge_outputs:
                        writer.add_scalar(f'loss_weights/{s}', mtl.edge_loss_wt[s].item(), epoch)

                    if cfg.model=='RGCN':
                        writer.add_histogram('val/latent_z', model.z, epoch)
                        
                    if cfg.model=='DRGCN':
                        writer.add_histogram('val/latent_z', model.z, epoch)
                        writer.add_histogram('val/latent_a', model.a, epoch)

                    if args.vis_graph or args.plot_traj:

                        for phase in ['train', 'test']:
                            samples, index = datasets[phase].first_sample_with_n_agents(5)
                            _, raw_data_dict = evaluate_model(cfg, model, [samples], device, return_raw_data=True, prediction_model=prediction_model)
                            raw_data = raw_data_dict[0]

                            pred_pos, prob = rank_predictions_over_top_k_modes(raw_data['pred_pos'], raw_data['ade'], top_k=5)
                            
                            obsv_graphs = raw_data['obsv_graphs']
                            gt_graphs = raw_data['gt_graphs']
                            if args.vis_graph:
                                gt_graphs.ndata['pred_pos'] = traj_to_graph_data(gt_graphs, traj=pred_pos, traj_ids=raw_data['comm_traj'])[0].squeeze(0)
                                pred_edges = edata_from_ndata(gt_graphs, gt_graphs.ndata['pred_pos'])
                                gt_graphs.edata['pred_dist'] = pred_edges['dist']
                                _, ax = network_draw(obsv_graphs, node_label='tid', show_edge_label=True, edge_label='dist')
                                _, ax = network_draw(gt_graphs, node_label='tid', show_edge_label=True, edge_label='dist', ax=ax)
                                _, ax = network_draw(gt_graphs, node_label='tid', show_edge_label=True, edge_label='pred_dist', ax=ax, 
                                                    pos_attr='pred_pos', node_shape='s')
                                ax.figure.savefig(f"{args.vis_graph_dir}{phase}_epoch{epoch}_frame{index}.png")

                            if args.plot_traj:
                                obsv_pos = get_ndata(raw_data['obsv_graphs'], 'pos', raw_data['comm_traj'])
                                gt_pos = get_ndata(gt_graphs, 'pos', raw_data['comm_traj'])
                                pred_pos = raw_data['pred_pos'].permute(1, 0, 2, 3)
                                plot_path(obsv_pos, gt_pos, pred_pos, K=10, ms=3, 
                                          save_dir=args.plot_traj_dir, min_obsv_len=1, min_pred_len=6, 
                                          fprefix=f"{phase}_epoch{epoch}_scene_{obsv_graphs.ndata['sid'][0].item()}",
                                          counter=obsv_graphs.ndata['fid'][-1].item()
                                          )

                if epoch>cfg.lr_reduce_epoch:
                    if isinstance(lr_scheduler, ReduceLROnPlateau) or isinstance(lr_scheduler, ReduceLROnPlateauWithWarmup):
                        lr_scheduler.step(h[cfg.lr_scheduler_metric][-1])
                    else:
                        lr_scheduler.step()

                if early_stopping.early_stop:
                    logger.info("Early stopping at epoch {}".format(epoch))
                    break

    except KeyboardInterrupt: # this is to prevent from accidental ctrl + c
        logger.info('-' * 89)
        logger.info('Exiting from training early because of KeyboardInterrupt')

    cfg.final_epoch = epoch
    if not args.debug and args.local_rank==0:
        #save metrics to csv
        args_params = ['user', 'trial', 'run', 'prefix', 'datetime', 'single_batch']

        cfg_params = ['data_dir', 'model', 'version', 'num_parameters', 'train_bs','test_bs',
                    'skip', 'min_seq_len', 'min_obsv_len', 'augment_prob',
                    'lr', 'lr_reduce_factor','epochs', 'final_epoch',
                    ]

        metrics = ['test_ade', 'test_fde', 'min_test_ade', 'min_test_fde','test_ade_best',
                    'test_fde_best', 'min_test_ade_best', 'min_test_fde_best']

        row_data = [vars(args)[k] for k in args_params] + [getattr(cfg, k) for k in cfg_params] + [h[k][-1] for k in metrics]

        save_to_excel(args.out_dir + 'log_results_trial_{}.xlsx'.format(args.trial),
                      row_data, header=args_params + cfg_params + metrics)


        # results = {k:v for k, v in zip(args_params + cfg_params + metrics, row_data)}
        # with open(args.out_path + '/results.json', 'w') as f:
        #     json.dump(results, f)

        with open(args.out_path +'/history.json', 'w') as fp:
            json.dump(h, fp)

        with open(args.out_path + '/test_results.txt', 'a+') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['{}:{:.3f}'.format(metric, h[metric][-1]) for metric in metrics])

        writer.close()

    return h


def main():

    args = parse_args()
    print(vars(args))

    cfg = Config.fromfile(args.config)

    # overwrite cfg dict from argparsers
    for k, v in vars(args).items():
        #if hasattr(cfg, k):
        setattr(cfg, k, v)

    print({k:v for k, v in cfg.serialize.items() if k!='standardization'})

    #setup seeds
    set_random_seed(cfg.seed)

    if args.distributed:

        print('Master -{}:{}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))

        # world_size = os.environ['nnodes'] * os.environ['nproc_per_node']
        args.world_size=int(os.environ['WORLD_SIZE'])
        print('world size:', args.world_size)

        args.local_rank = int(os.environ["LOCAL_RANK"]) # used to indicate which gpu to use
        print('local rank:', args.local_rank)

        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank % num_gpus)
        device = torch.device("cuda:{}".format(args.local_rank%num_gpus))

        torch.distributed.init_process_group(backend="nccl", init_method='env://', world_size=args.world_size, rank=args.local_rank)

    else:
        device = torch.device("cuda:{}".format(args.gpu_id))
        args.world_size = 1


    # cfg.train_bs = cfg.train_bs * cfg.world_size
    # if cfg.auto_scale_lr:
    #     # cfg.lr = cfg.lr * num_gpus/8 # https://github.com/open-mmlab/mmdetection3d/blob/master/tools/train.py
    #     base_bs = 16
    #     cfg.lr  = (cfg.train_bs/base_bs) * cfg.lr


    if cfg.model == 'STDec':
        run_path = '_'.join([f'run-{args.run}',
                            f'{cfg.model}',
                            # f'{cfg.net.type}',
                            f'{cfg.net.layer}',
                            f'h{cfg.hidden_dim}',
                            f'l{cfg.num_layers}',
                            # f'{cfg.version}',
                            f'K{cfg.K}',
                            f'K_eval{cfg.K_eval}',
                            f'bs{cfg.train_bs}',
                            # f'{cfg.node_loss}{cfg.node_loss_wt}',
                            # f"{cfg.edge_loss}{cfg.edge_loss_wt[0]}"
                            # '_'.join(cfg.node_inputs),
                            'pred_' + '_'.join([s+str(cfg.node_loss_wt.serialize[s]) for s in cfg.node_outputs])
                            #'_'.join(cfg.edge_inputs),
                            #'_'.join(cfg.node_types),
                            ])
    else:
        run_path  = '_'.join([f'run-{args.run}',
                            f'{cfg.model}',
                            #f'{cfg.net.type}',
                            f'{cfg.net.layer}',
                            # f'z{cfg.z_dim}',
                            # f'a{cfg.a_dim}',
                            f'eh{cfg.enc_hdim}',
                            f'el{cfg.enc_layers}',
                            f'K{cfg.K}',
                            f'K_eval{cfg.K_eval}',
                            f'bs{cfg.train_bs}',
                            f'skip{cfg.skip}',
                            # f'{cfg.traj_loss}',
                            # f'{cfg.traj_dec.type}',
                            # '_'.join(cfg.node_inputs),
                            # '_'.join(cfg.node_types),
                            'pred_' + '_'.join([s+str(cfg.node_loss_wt.serialize[s]) for s in cfg.node_outputs])
                            # f'{cfg.version}',
                            ])

    if len(cfg.edge_outputs)>0:
        run_path += '_edge_' + '_'.join([s+str(cfg.edge_loss_wt.serialize[s]) for s in cfg.edge_outputs])
        
    if args.preprocess:
        run_path += '_preprocess'
        
    if cfg.fov_graph:
        run_path += '_fov_graph'

    if args.single_batch:
        run_path += '_single_batch'
        
    if len(args.prefix)>0:
        run_path += f'_{args.prefix}'

    if cfg.include_lane:
        run_path += f"_lane{INTERACTION_RADIUS['LANE', 'VEHICLE']}"
    
    # run_path += f"_{getpass.getuser()}"

    args.run_path = run_path
    args.user = getpass.getuser()
    args.datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    args.out_dir = f'../out/{cfg.dataset}'
    args.out_path = f'{args.out_dir}/trial_{args.trial}/{run_path}'
    args.root_path = os.path.dirname(os.path.realpath(__file__))

    start = time.time()



    if cfg.dataset=='pedestrians':
        dset_results = defaultdict(dict)
        tmp_out_path = args.out_path
        for dset_name in cfg.dset_names:
            cfg.dset = dset_name
            cfg.data_dir = '../datasets/pedestrians/' + dset_name 

            if cfg.model == 'STDec':
                if cfg.trajdec=='ConstVel':
                    prediction_model = ConstantVelocity(sec_from_now=np.round(cfg.pred_seq_len * cfg.dt, 3), sampled_at=1/cfg.dt)
                    prediction_model_name = 'ConstVel'

                elif cfg.trajdec.startswith('run'):
                    ckpt_dir = f"{args.out_dir}/trial_{args.trial}/{cfg.trajdec}"
                    prediction_model, _, pred_cfg = get_model_from_src('.', osp.join(ckpt_dir, dset_name), 'min_test_ade_best', device)
                    prediction_model_name = f"{pred_cfg['model']}_K{pred_cfg['K']}" 
                else:
                    raise Exception('Traj dec model not valid!')

            args.out_path = f"{tmp_out_path}_{prediction_model_name}/{dset_name}"
            
            args = preprare_and_backup(args, cfg)

            print(f'****** Running {dset_name} *********')
            results = run(args, cfg, device, prediction_model)

            for m in ['min_test_ade', 'min_test_fde', 'min_test_ade_best', 'min_test_fde_best']:
                dset_results[m][dset_name] = np.round(results[m][-1], 3)

        avg_results = {}
        for m, result in dset_results.items():
            print(m, result)
            avg_results['avg_'+m] = np.average([v for k, v in result.items()]).round(2)
        
        avg_results.update(dset_results)
        print(f"Averge ADE/FDE-1:{avg_results['avg_min_test_ade']}/{avg_results['avg_min_test_fde']}")
        print(f"Averge ADE/FDE-{cfg.K_eval}:{avg_results['avg_min_test_ade_best']}/{avg_results['avg_min_test_fde_best']}")
        
        if not args.debug:
            with open( tmp_out_path + '/ade_fde_results.txt', 'a+') as f:
                json.dump(avg_results, f)

    elif cfg.dataset =='nuscenes':
        
        if cfg.model == 'STDec':
            prediction_model = get_model_from_src(cfg.trajdec_trial + cfg.trajdec_run + '/', 'min_test_ade_best', device)[0]
            args.out_path += '_' + prediction_model.__class__.__name__

        args = preprare_and_backup(args, cfg)

        run(args, cfg, device, prediction_model)

    print('Total training time:', time.time()-start)

    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__=='__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('Exiting from training early because of KeyboardInterrupt')
        sys.exit()
    except Exception:
        traceback.print_exc()