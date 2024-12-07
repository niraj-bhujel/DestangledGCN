#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 00:27:03 2020

@author: dl-asoro
"""
import argparse
        
def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--config', type=str, default='./configs/pedestrians_gcn.py')
    parser.add_argument('--dset_names', nargs='+', default=['eth', 'hotel', 'univ', 'zara1', 'zara2'])

    # data
    parser.add_argument('--preprocess', action='store_true', default=False)
    parser.add_argument('--num_process',type=int, default=0)

    # Training
    parser.add_argument('--seed', type=int, default=4309)
    parser.add_argument('--fov_graph', action='store_true', default=False, help='If fov graph, connect all spatial nodes with fov constraints')

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--vis_graph', action='store_true', default=False)

    parser.add_argument('--resume_training', action='store_true', default=False)
    parser.add_argument('--resume_epoch', type=int, default=155, 
                        help='initial_epoch - 1')

    parser.add_argument('--initial_epoch', type=int, default=1, 
                        help='Start epoch number')

    parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=True)
    parser.add_argument('--no_overwrite', dest='overwrite', action='store_false', default=False)
    
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of workers (0 - max_num_cpu), also >0 freeze distrbuted dataparallel model")

    # GPU
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='Default gpu to use')
    parser.add_argument('--num_gpus', type=int, default=1, 
                        help='Number of gpus to use (if distrbuted use all gpus)')
    parser.add_argument('--gpu_memory', type=int, default=0,
                        help='Minimum GPU memory (MB)')
    parser.add_argument('--device_type', default='cuda', 
                        help='cuda or cpu')

    # Distributed training
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--launcher', default='none', choices=['none', 'pytorch_launch', 'pytorch_elastic'],
                        help='Launcher for distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank (set by torch distributed)')
    parser.add_argument('--auto_scale_lr', action='store_true',
                        help='Auto scale learning rate with batch size')


    #--logging
    parser.add_argument('--trial', type=int, default=5, 
                        help='tial number for trianing')
    parser.add_argument('--run', type=int, default=0, 
                        help='Run number for trianing')
    parser.add_argument('--prefix', default='', 
                        help='Prefix for trianing')
    parser.add_argument('--debug', action='store_true', default=False, 
                        help="disable all disk writing processes.")
    parser.add_argument('--single_batch', action='store_true', default=False, help='Use single batch data loader, useful for debugging')

    parser.add_argument('--plot_traj', action='store_true', default=False)
    parser.add_argument('--verbose', type=int, default=1, 
                        help='Display outputs')

    args = parser.parse_args()

    return args
    

if __name__=='__main__':
    args = parse_argument()

    print(args.node_type, args.pred_states)
