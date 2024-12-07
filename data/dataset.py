#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:54:28 2021

@author: dl-asoro
"""
import os
import time
import pickle
import math
import multiprocessing as mp
import numpy as np

from tqdm import tqdm
from glob import glob

import dgl
import torch
from torch.utils.data import Dataset

from data.states import *
from data.data_augmentation import augment_graph_data
from data.preprocess_utils import seq_to_st_graph, seq_to_fov_graph
from data.positional_encoding import graph_positional_encoding

from utils.graph_utils import split_graph, filter_node_types
from utils.misc import create_new_dir

class BaseDataset(Dataset):
    def __init__(self, **kwargs):

        super(BaseDataset, self).__init__()
        
        self.__dict__.update(kwargs)

    def __len__(self):
        return len(self.graphs_list)

    def __getitem__(self, index):

        g = self.graphs_list[index]

        g = augment_graph_data(g, probability=self.aug_prob, dt=self.dt, obsv_len=self.obsv_len)
        
        # g.edata['temporal_mask'] = 1 - g.edata['spatial_mask']
        
        g1, g2 = split_graph(g, split_idx=self.obsv_len-1)

        return (g, g1, g2)

    def collate(self, samples):
        # The input samples is a list of tuples graphs [(seq_graphs, obsv_graphs, trgt_graphs)].
        seq_graphs, obsv_graphs, trgt_graphs = map(list, zip(*samples))
        batched_seq_graphs = dgl.batch(seq_graphs, NODE_ATTRS, EDGE_ATTRS)
        batched_obsv_graphs = dgl.batch(obsv_graphs, NODE_ATTRS, EDGE_ATTRS)
        batched_trgt_graphs = dgl.batch(trgt_graphs, NODE_ATTRS, EDGE_ATTRS)
        return batched_seq_graphs, batched_obsv_graphs, batched_trgt_graphs

    def first_sample(self, ):
        g = self.graphs_list[0]
        g.edata['temporal_mask'] = 1 - g.edata['spatial_mask']
        g1, g2 = split_graph(g, split_idx=self.obsv_len-1)
        return (g1, g2)

    def first_sample_with_n_agents(self, min_agents=5):

        index = 0
        while True:
            
            g = self.graphs_list[index]
            index += 1
            if len(g.ndata['tid'].unique())>min_agents:
                break

        g.edata['temporal_mask'] = 1 - g.edata['spatial_mask']
        g1, g2 = split_graph(g, split_idx=self.obsv_len-1)

        return self.collate([(g1, g2)]), index-1


class PedestriansDataset(BaseDataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir, phase, **kwargs):

        super(PedestriansDataset, self).__init__(**kwargs)
        
        self.data_dir = data_dir
        self.phase = phase
        
        self.seq_len = self.obsv_len + self.pred_len
                
        # self.data_dir = '../datasets/pedestrians/'+dset_name+'/'
        self.dset_name = os.path.basename(os.path.normpath(self.data_dir))
        self.graph_data_dir = os.path.join(os.getenv('ROOT_DIR', '../'), f'graphs_dataset/pedestrians/{self.dset_name}/{self.phase}')
        fprefix = f"obsv{self.obsv_len}_pred{self.pred_len}_skip{self.skip}_minseq{self.min_seq_len}"
        if self.fov_graph:
            fprefix += '_fov_graph'
        else:
            fprefix += '_st_graph' 
        
        self.pickle_file = self.graph_data_dir + '/' + fprefix + '.pkl'
        if not self.preprocess:
            print('Loading:', self.pickle_file)
            try:
                with open(self.pickle_file, 'rb') as f: 
                    data = pickle.load(f)
                    self.graphs_list = data['seq_graphs']
                    
            except Exception as e:
                print(e)
                self._preprocess()
        else:
            self._preprocess()
    

    def _preprocess(self, ):
        from data.preprocess_pedestrians import preprocess_sequence

        processed_sequence = preprocess_sequence(self.data_dir,
                                                self.phase, 
                                                min_seq_len=self.min_seq_len, 
                                                skip=self.skip, 
                                                min_peds=self.min_agents,
                                                padding=not self.fov_graph)
        
        #Convert to Graphs
        print("\nPreparing {} {} graphs ...".format(self.dset_name, self.phase,))
        start_time = time.time()
        self.graphs_list = []
        # pbar = tqdm(total=len(processed_sequence), position=0) 
        for sequence in processed_sequence:
            # pbar.update(1)
            if self.fov_graph:
                g = seq_to_fov_graph(sequence, self.interaction_radius)
            else:
                g = seq_to_st_graph(sequence, self.interaction_radius, seq_len=self.seq_len)
            
            if g.number_of_nodes()>1 and g.number_of_edges()>1:
                self.graphs_list.append(g)
            
        if not os.path.exists(self.graph_data_dir):
            os.makedirs(self.graph_data_dir)
            
        with open(self.pickle_file, 'wb') as f: #%%
   
            pickle.dump({'seq_graphs':self.graphs_list}, f)
            
        print('\nFinished preparing {} graphs in {:.1f}s'.format(self.phase, time.time()-start_time))

class NuScenesDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir, phase, version, node_type, interaction_radius, 
                 obsv_len=8, pred_len=12, min_obsv_len=2, min_seq_len=12, 
                 skip=1, min_ped=1, dt=0.5, aug_prob=0.5, pos_enc=None, pos_enc_dim=16,
                  include_robot=False, include_lane=False, preprocess=False, **kwargs):

        super(NuScenesDataset, self).__init__()
        # self.dataset = dataset
        self.phase = phase
        self.version = version

        self.obsv_len = obsv_len
        self.pred_len = pred_len
        self.seq_len = self.obsv_len + self.pred_len
        self.skip = skip
        self.dt = dt
        self.aug_prob = aug_prob

        self.min_ped = min_ped
        self.min_obsv_len = min_obsv_len
        self.min_seq_len= min_seq_len
        self.preprocess = preprocess
        
        # self.node_type = node_type
        self.node_attrs = ['ntx', 'tid', 'cid', 'sid', 'nid', 'fid', 'pos', 'rel', 'vel', 'acc', 'yaw', 'dir', 'vnorm', 'anorm']
        self.edge_attrs = ['dist', 'diff', 'spatial_mask']

        self.include_robot = include_robot
        self.include_lane = include_lane
        self.interaction_radius = interaction_radius

        self.pos_enc = pos_enc
        self.pos_enc_dim = pos_enc_dim
        
        self.data_dir = data_dir
        # self.data_dir = os.path.join(os.getenv('ROOT_DIR', '..'), '/datasets/nuscenes/processed_data')

        # self.dset_name = os.path.basename(os.path.normpath(self.data_dir))
        self.graph_data_dir = os.getenv('ROOT_DIR', '../') + f'graphs_dataset/nuscenes/{self.phase}/'
        self.pickle_file = 'obsv{}_minobsv{}_pred{}_skip{}_minseq{}_{}_{}_idist_{}'.format(self.obsv_len,
                            self.min_obsv_len,
                            self.pred_len,
                            self.skip,
                            self.min_seq_len,
                            self.version,
                            self.node_type,
                            str(int(self.interaction_radius[node_type, node_type]))
                            )
        
        if self.include_robot:
            self.pickle_file += '_robot'
        if self.include_lane:
            self.pickle_file += '_lane_dist{}'.format(int(self.interaction_radius['VEHICLE', 'LANE']))
            
        if not self.preprocess:
            try:
                print('Loading {} graphs from ... {}'.format(self.phase, self.pickle_file))
                with open(self.graph_data_dir + self.pickle_file + '.pkl', 'rb') as f: 
                    data = pickle.load(f)
                    self.graphs_list = data['seq_graphs']
                    self.token_map = data['token_map']
                    self.scene_pos_mean = data['scene_pos_mean']
                    
            except Exception as e:
                print(e)
                self._preprocess()
        else:
            self._preprocess()

        # if pos_enc is not None:
        # print(f"Adding positional encoding ... ")
        if pos_enc:
            for g in self.graphs_list:
                graph_positional_encoding(g, pos_enc_dim, 'rand_walk')
            
    def __len__(self):
        return len(self.graphs_list)

    def __getitem__(self, index):

        g = self.graphs_list[index]

        g = augment_graph_data(g, probability=self.aug_prob, dt=self.dt, obsv_len=self.obsv_len)

        g1, g2 = split_graph(g, split_idx=self.obsv_len-1)

        return (g1, g2)
    
    def collate(self, samples):
        # The input samples is a list of tuples graphs [(seq_graphs, obsv_graphs, trgt_graphs)].
        obsv_graphs, trgt_graphs = map(list, zip(*samples))
        # batched_seq_graphs = dgl.batch(seq_graphs, self.node_attrs, self.edge_attrs)
        batched_obsv_graphs = dgl.batch(obsv_graphs, self.node_attrs, self.edge_attrs)
        batched_trgt_graphs = dgl.batch(trgt_graphs, self.node_attrs, self.edge_attrs)
        return batched_obsv_graphs, batched_trgt_graphs
    

    def _preprocess(self, ):
        from preprocess_sequence import preprocess_sequence
        processed_sequence, token_map, scene_pos_mean = preprocess_sequence(phase=self.phase, 
                                                                            data_dir=self.data_dir, 
                                                                            # node_type_list=NODE_TYPES,
                                                                            version=self.version,
                                                                            obsv_len=self.obsv_len, 
                                                                            seq_len=self.seq_len, 
                                                                            min_obsv_len=self.min_obsv_len,
                                                                            min_seq_len=self.min_seq_len, 
                                                                            skip=self.skip, 
                                                                            min_ped=self.min_ped, 
                                                                            dt=self.dt,
                                                                            include_robot=self.include_robot, 
                                                                            include_lane=self.include_lane,
                                                                            node_type=self.node_type)
        
        #Convert to Graphs
        print("\nPreparing {} graphs ...({} graphs)".format(self.phase, len(processed_sequence)))
        start_time = time.time()
        self.token_map = token_map
        self.scene_pos_mean = scene_pos_mean
        self.graphs_list = []
        # pbar = tqdm(total=len(processed_sequence), position=0) 
        for sequence in processed_sequence:
            # pbar.update(1)
            g = seq_to_st_graph(sequence, self.interaction_radius, seq_len=self.seq_len)
            if g.number_of_nodes()>1 and g.number_of_edges()>1:
                self.graphs_list.append(g)
                
        # pbar.close()
        
        if not os.path.exists(self.graph_data_dir):
            os.makedirs(self.graph_data_dir)
            
        with open(self.graph_data_dir + self.pickle_file + '.pkl', 'wb') as f: #%%
            pickle.dump({'seq_graphs':self.graphs_list, 'token_map': self.token_map, 'scene_pos_mean':self.scene_pos_mean}, f)
            
        print('\nFinished preparing {} graphs in {:.1f}s'.format(self.phase, time.time()-start_time))


class WaymoDataset(Dataset):
    """Dataloder for the Trajectory datasets ( for multiple node type) """
    def __init__(self, data_dir, phase, version, node_types, interaction_radius, 
        obsv_len=3, pred_len=16, min_obsv_len=2, min_seq_len=2, skip=1, min_agents=1, dt=0.5, aug_prob=0.5,
        include_lane=False, preprocess=False, preprocess_graphs=False, num_process=10, clean_dir=False):

        super(WaymoDataset, self).__init__()
        
        self.data_dir = data_dir
        # self.dataset = dataset
        if phase == 'train' or phase=='test':
            self.phase = phase + 'ing'
        else:
            self.phase = 'validation'

        self.version = version

        self.obsv_len = obsv_len
        self.pred_len = pred_len
        self.seq_len = self.obsv_len + self.pred_len
        self.skip = skip
        self.dt = dt
        self.aug_prob = aug_prob

        self.min_agents = min_agents
        self.min_obsv_len = min_obsv_len
        self.min_seq_len= min_seq_len
        self.preprocess = preprocess
        
        self.node_types = node_types
        
        self.include_lane = include_lane
        self.interaction_radius = interaction_radius
        # self.preload = preload
        self.clean_dir = clean_dir
        self.num_process = min(num_process, mp.cpu_count())
        self.graphs_lists = []

        self.prefix = 'obsv{}_pred{}_minlen{}_skip{}_{}'.format(#'_'.join([s.lower() for s in self.node_types]), 
                                                            self.obsv_len,
                                                            self.pred_len,
                                                            self.min_seq_len,
                                                            self.skip,
                                                            self.version)
        if self.include_lane:
            self.prefix += '_lane'

        # self.data_dir = '/run/user/1000/gvfs/smb-share:server=access.serc.acrces2.shared-svc.local,share=i2r/home/bhujeln1/waymo-od/tf_example'
        # self.data_dir = '../datasets/waymo/processed_data/'
        
        root_dir = os.getenv('ROOT_DIR', '..')
        self.graph_data_dir = f'{root_dir}/datasets/waymo/graphs/{self.phase}/{self.prefix}'

        self.preprocess_dir = f'{root_dir}/datasets/waymo/processed_sequence/{self.phase}/{self.prefix}' 
            

        if preprocess:
            self.graphs_list = self._preprocess()

        elif preprocess_graphs:
            self.graphs_list = self._prepocess_graphs()

        else:
            self.graphs_list = sorted(glob(os.path.join(self.graph_data_dir, "*.bin")))
            print(f'{len(self.graphs_list)} {phase} graphs!!')

            if not len(self.graphs_list)>0:
                try:
                    self.graphs_list = self._prepocess_graphs()
                
                except Exception as e:
                    print(e)
                    raise Exception('Unable to create graphs..please preproces sequences first')           
            

    def _preprocess(self, ):

        from data.preprocess_waymo import preprocess_sequence, NUM_GT_RECORDS
        from multiprocessing import Process, Manager, Value

        print(f'Preprocessing {self.phase} sequences to {self.preprocess_dir}')
        start = time.time()


        tf_records = sorted(glob(os.path.join(self.data_dir, self.phase, "*")))
        # use only 60% for faster training, at least one scene
        if self.phase=='training' or self.phase=='validation':
            # max_record = np.maximum(1, int(self.version.split('_')[-1]) * NUM_GT_RECORDS[self.phase] * 0.01).astype(int)
            max_record = 1
            tf_records = tf_records[:int(max_record)]
            print(f"Processing only {len(tf_records)} {self.phase} records ", )

        kwargs = dict(data_dir=self.data_dir, 
                    phase=self.phase, 
                    version=self.version,
                    obsv_len=self.obsv_len, 
                    seq_len=self.seq_len, 
                    min_seq_len=self.min_seq_len, 
                    skip=self.skip, 
                    min_agents=self.min_agents, 
                    dt=self.dt,
                    # aug_prob=self.aug_prob,
                    include_lane=self.include_lane,
                    save_path=create_new_dir(self.preprocess_dir, clean=self.clean_dir),
                    tf_records=tf_records,
                    )

        if self.num_process>1:
    
            num_splits = math.ceil(len(tf_records)/self.num_process)
            tf_records_splits = [tf_records[num_splits*n:num_splits*(n+1)] for n in range(self.num_process)]
            pools = []
            for pid in range(0, self.num_process):
                kwargs.update(pid=pid, tf_records=tf_records_splits[pid])
                p = mp.Process(target=preprocess_sequence, kwargs=kwargs)
                pools.append(p)
                p.start()
                time.sleep(1.)
            
            for p in pools:
                p.join()
        else:
            preprocess_sequence(**kwargs)

        print(f"Finished preprocessing {self.phase} sequences in :{time.time()-start:.3f} secs")

        graphs = self._prepocess_graphs() # multi-processing for val set throwing error

        return graphs

    def _prepocess_graphs(self,):
        
        seq_list = sorted(glob(os.path.join(self.preprocess_dir, "*.bin")))
        print(f'\nProcessing graphs from {self.preprocess_dir} ({len(seq_list)} sequences)')

        graphs = []
        
        save_dir = create_new_dir(self.graph_data_dir, self.clean_dir)

        for file in tqdm(seq_list):
            # print(f)
            try:
                with open(file, 'rb') as f:
                    seq = pickle.load(f)
                # seq = torch.load(f)
                g = seq_to_st_graph(seq, self.interaction_radius, seq_len=self.seq_len)
                
            except Exception as e:
                print(e)
                # raise Exception "failed to convert sequence to graph"
                continue
            
            graphs.append(g)

            with open(f"{self.graph_data_dir}/{os.path.basename(file)}", 'wb') as f:
                pickle.dump(g, f)


        # re-asign global_tid as there is overlap during multi-processing
        global_tid = 0
        for g in graphs:
            agents_id = g.ndata['nid'].unique()
            for nid in agents_id:
                global_tid += 1
                g.ndata['tid'][g.ndata['nid']==nid] = global_tid

        print(f'Finished creating {len(graphs)} {self.phase} graphs (saved to {self.graph_data_dir})')

        return graphs

    def __len__(self):
        
        return len(self.graphs_list)

    def __getitem__(self, index):

        g = self.graphs_list[index]
        if not isinstance(g, dgl.DGLHeteroGraph):
            with open(g, 'rb') as f:
                g = pickle.load(f)

        # g = augment_graph_data(g, probability=self.aug_prob, dt=self.dt)

        # random drop spatial edges
        if self.phase=='train':
            temporal_edges = torch.where(g.edata['spatial_mask'].flatten()==0)[0]
            spatial_edges = torch.where(g.edata['spatial_mask'].flatten()==1)[0]
            num_drop_edges = int(0.4*len(spatial_edges))

            if num_drop_edges>1:
                spatial_edges = torch.randperm(len(spatial_edges))[:num_drop_edges] # drop 40% of spatial edges
            
            g = dgl.edge_subgraph(g, edges=torch.cat([spatial_edges, temporal_edges]))

        g1, g2 = split_graph(g, split_idx=self.obsv_len-1)

        return (g1, g2)

    def collate(self, samples):
        # The input samples is a list of tuples graphs [(seq_graphs, obsv_graphs, trgt_graphs)].
        g1, g2 = map(list, zip(*samples))

        # batched_seq_graphs = dgl.batch(seq_graphs, self.node_attrs, self.edge_attrs)
        g1 = dgl.batch(g1, self.node_attrs, self.edge_attrs)
        g1 = filter_node_types(g1, self.node_types, NODE_TYPES)

        g2 = dgl.batch(g2, self.node_attrs, self.edge_attrs)
        g2 = filter_node_types(g2, self.node_types, NODE_TYPES)

        return g1, g2