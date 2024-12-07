import os
import sys
import json
import pickle
import os.path as osp
import torch
from utils.misc import import_reload
import numpy as np
def save_ckpt(model, optimizer, scheduler, history, epoch, ckpt_dir, ckpt_name='full_model'):
    state = {
        'last_epoch': epoch,
        'history': history,
        'model_state': model.state_dict(),
        'optimizer_state': {k:optim.state_dict() for k, optim in optimizer.items()},
        'scheduler_state':{k:schd.state_dict() for k, schd in scheduler.items()},
        }
    torch.save(state, ckpt_dir+ ckpt_name + '.pth')

def load_ckpt(model, optimizer, scheduler, ckpt_dir, ckpt_name='full_model'):
    checkpoint = torch.load(ckpt_dir + ckpt_name + '.pth')
    last_epoch = checkpoint['last_epoch']
    history = checkpoint['history']
    model.load_state_dict(checkpoint['model_state'], strict=True)

    for k, _ in optimizer.items():
        optimizer[k].load_state_dict(checkpoint['optimizer_state'][k])
        scheduler[k].load_state_dict(checkpoint['scheduler_state'][k])

    return model, optimizer, scheduler, history, last_epoch

def save_ckpt_simple(model, optimizer, scheduler, history, epoch, ckpt_dir, ckpt_name='full_model'):
    state = {
        'last_epoch': epoch,
        'history': history,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state':scheduler.state_dict(),
        }
    torch.save(state, osp.join(ckpt_dir, ckpt_name + '.pth'))

def load_ckpt_simple(model, optimizer, scheduler, ckpt_dir, ckpt_name='full_model'):
    checkpoint = torch.load(osp.join(ckpt_dir, ckpt_name + '.pth'))
    last_epoch = checkpoint['last_epoch']
    history = checkpoint['history']
    model.load_state_dict(checkpoint['model_state'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])

    return model, optimizer, scheduler, history, last_epoch


def get_model_from_ckpt(args, model_params, ckpt_path, gnn_model, device, map_location='cuda:0', strict=True):

    print('Loading model ... ', ckpt_path)
    if 'full' in os.path.basename(ckpt_path):
        my_model = torch.load(ckpt_path)

    elif 'min' in ckpt_path:
        my_model = gnn_model(args.model, model_params, args)
        my_model.load_state_dict(torch.load(ckpt_path, device), strict=strict)
    else:
        raise Exception('Incorrect checkpoint!')
        
    my_model = my_model.to(device)
    return my_model

def get_model_from_src(src_dir, ckpt_dir=None, ckpt_name='min_test_ade_best', device='cuda:0', strict=True):

    # imp will load only one module, not the other modules required by model from the path
    # foo = imp.load_source('module.name', src_dir + 'src/model/model.py')
    # print('model imported from:', inspect.getfile(foo.gnn_model))

    src_path = osp.join(src_dir, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from model.models import gnn_model as models 
    # import_reload('model.models') # import_reload will overwrite previously imported modules from model.models i.e. will create problem if called from train.py 

    print('get model from src:', sys.modules['model.models'])
    sys.path.remove(src_path) # remove source after importing

    if ckpt_dir is None:
        ckpt_dir = src_dir
    
    with open(ckpt_dir + '/args.pkl', 'rb') as f: 
        args = pickle.load(f)
    
    with open(ckpt_dir + '/cfg.json', 'rb') as f:
        cfg_dict = json.load(f)

    #print(cfg_dict)
    my_model = models(cfg_dict['model'], cfg_dict)
    # model_attributes(my_model, 1)
    print(my_model)


    ckpt_path = osp.join(ckpt_dir, ckpt_name + '.pth')
    print('Loading model', ckpt_path)

    model_state = torch.load(ckpt_path, map_location=device)
    # if args.distributed:
    model_state = {k.replace('module.', ''):v for k, v in model_state.items()}
    my_model.load_state_dict(model_state, strict=strict)
        
    my_model = my_model.to(device)
    
    return my_model, args, cfg_dict

def model_attributes(model, verbose=0):
    attributes = {k:v for k, v in model.__dict__.items() if not k.startswith('_')}
    
    if verbose>0:
        print({k:attributes[k] for k in sorted(attributes.keys()) if k!='net_params'})
        
    return attributes

def model_parameters(model, verbose=0):
    if verbose>0:
        print('{:<30} {:<10} {:}'.format('Parame Name', 'Total Param', 'Param Shape'))
    total_params=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if verbose>0:
                print('{:<30} {:<10} {:}'.format(name, param.numel(), tuple(param.shape)))
            total_params+=param.numel()
    print('Total Trainable Parameters :{:<10}'.format(total_params))
    return total_params

# use computation graph to find all contributing tensors
def get_contributing_params(y, top_level=True):
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
            print(f.shape)
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)

def non_contributing_params(model, outputs=[]):
    contributing_parameters = set.union(*[set(get_contributing_params(out)) for out in outputs])
    all_parameters = set(model.parameters())
    non_contributing_parameters = all_parameters - contributing_parameters
    if len(non_contributing_parameters)>0:
        print([p.shape for p in non_contributing_parameters])  # returns the [999999.0] tensor
    return sum([np.product(list(p.shape)) for p in non_contributing_parameters])

def init_weights_kaiming(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        torch.nn.init.zeros_(m.bias.data)

def init_weights_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=1.4142135623730951) # gain for relu
        torch.nn.init.zeros_(m.bias.data)

def exponential_weight(step, init_value=1e-3, total_steps=300, decay_steps=200):
    '''
    create a exponentially increasing values, from init_value to 1. 
    '''
    return init_value ** ((total_steps - step) / decay_steps)