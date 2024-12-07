NODE_INFO = ['nid', 'tid', 'cid', 'sid', 'htl', 'ftl', 'dt']
NODE_STATES = ['ntx', 'fid', 'pos', 'rel', 'vel', 'vnorm', 'dir', 'speed', 'acc', 'anorm', 'yaw', 'hed', 'goal',]

NODE_ATTRS =  NODE_INFO + NODE_STATES
EDGE_ATTRS = ['dist', 'diff', 'dir', 'spatial_mask', 'temporal_mask']

STATE_DIMS = {
    'ntx': 1,
    'nid': 1,
    'tid': 1,
    'fid': 1,
    'htl': 1,
    'ftl': 1,
    'dt': 1,
    'pos': 2,
    'rel': 2,
    'vel': 2,
    'acc': 2,
    'dir': 2,
    'rnorm':1,
    'anorm':1,
    'vnorm':1,
    'speed': 1, 
    'hed': 2,
    'goal': 2,
    'yaw': 1,
    'dist': 1,
    'diff': 2,
    'dir': 2,
    'init_vel': 2,
    'spatial_mask': 1,
    'temporal_mask': 1,
}

NODE_TYPES = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'LANE']# ordering is important for waymo since types are 1, 2, 3-> vehicle, pedestiran, cyclist, 

INTERACTION_RADIUS = {
    ('PEDESTRIAN', 'PEDESTRIAN'): 5.0,
    ('PEDESTRIAN', 'CYCLIST'): 10.0,
    ('PEDESTRIAN', 'VEHICLE'): 10.0,
    ('PEDESTRIAN', 'LANE') : 0.0,

    ('CYCLIST', 'CYCLIST'): 10.0,
    ('CYCLIST', 'VEHICLE'): 10.0,
    ('CYCLIST', 'LANE'): 0.0,

    ('VEHICLE', 'VEHICLE'): 10.0,
    ('VEHICLE', 'LANE'): 5.0,

    ('LANE', 'LANE'): 0.0,
    
}

QLEVELS = {
    'ntx': None,
    'nid': None,
    'tid': None,
    'fid': None,
    'htl': None,
    'ftl': None,
    'dt': None,
    'pos': 0.05,
    'rel': 0.05,
    'vel': 0.01,
    'acc': 0.01,
    'rnorm': 0.02,
    'anorm': 0.01,
    'vnorm': 0.01,
    'speed': 0.01,
    'dist': 0.01,
    'diff': 0.05,
    'goal': 0.05,
    'dir': 0.2,
    'hed': 0.2,
    'yaw': 0.2,
    'spatial_mask': None,
    'temporal_mask': None,
}
