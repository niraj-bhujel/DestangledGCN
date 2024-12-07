#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:21:14 2020

@author: dl-asoro
"""
import os
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.cbook as cb
import matplotlib.pylab as pylab
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
from matplotlib.patches import FancyArrowPatch

from copy import deepcopy

# line_colors = ['#375397', '#F05F78', '#80CBE5', '#ABCB51', '#C8B0B0', '#5535b8']
# line_colors = ['#ff0000', '#0000ff', '#008000', '#ffff00', '#ffa500', '#4b0082',] # red, blue, green , yellow, orange, indigo, violet
line_colors = ['#ff0000', '#10b798', '#dc417f', '#172774', '#ff0080', '#f47d33', '#9772FB', '#9d0208', '#04724d', '#6FDFDF']

def plot_color(colors):
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='datalim')
    r = 0.2
    cx = 0
    for i, color in enumerate(colors):
        
        circle = plt.Circle((cx, 0), r, color=color, alpha=0.7)
        ax.add_patch(circle)
        cx += r*2
        
    circle = plt.Circle((cx, 0), r, color='#cccccc')
    ax.add_patch(circle)
    ax.set_xlim([0, cx])
# plot_color(line_colors)

#%%
        
def get_color(idx, colors=None):
    if colors is not None:
        return colors[idx%len(colors)]
    idx = idx * 3
    color = [(37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255]
    color = [c/255 for c in color]
    return color

def network_draw(g, show_node_label=True, node_label='nid', show_edge_label=False, edge_label='dist', 
    pos_attr='pos', edge_attr='dist', node_size=200, font_size=8, show_legend=False, show_direction=False, 
    pad=(0, 0, 0, 0), extent=None, save_dir=None, fprefix=None, frame=None, counter=0, ax=None,  **kwargs):
    '''
    Parameters
    ----------
    g : TYPE
        DESCRIPTION.
    show_node_label : TYPE, optional
        DESCRIPTION. The default is False.
    show_edge_label : TYPE, optional
        DESCRIPTION. The default is False.
    edge_label : TYPE, can also be 'id' after converting dgl graph 'g' to networkx graph 'G'.
                    The id ordering could be different from original 'g'
        DESCRIPTION. The default is 'dist'.
    pos_attr : TYPE, optional
        DESCRIPTION. The default is 'pos'.
    node_size : TYPE, optional
        DESCRIPTION. The default is 300.
    show_legend : BOOL, if True, show the node number and corresponding tid 
        DESCRIPTION. The default is False.
    save_dir : TYPE, optional
        DESCRIPTION. The default is None.
    fprefix : TYPE, optional
        DESCRIPTION. The default is None.
    counter : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    '''
    # overwride from kwargs
    show_node_label = kwargs.get('show_node_label', show_node_label)
    show_edge_label= kwargs.get('show_edge_label', show_edge_label)
    show_legend = kwargs.get('show_legend', show_legend)
    frame = kwargs.get('frame', frame)
    counter = kwargs.get('counter', counter)
    min_margin = kwargs.get('min_margin', 0)
    extent = kwargs.get('extent', extent)
    save_dir = kwargs.get('save_dir', save_dir)
    fig_name = kwargs.get('fig_name', '')
    figsize = kwargs.get('figsize', (12, 8))
    limit_axes = kwargs.get('limit_axes', False)
    axis_off = kwargs.get('axis_off', False)
    dtext = kwargs.get('dtext', None)
    
    # node properties
    node_size = kwargs.get('node_size', node_size)
    node_shape = kwargs.get('node_shape', 'o')
    alpha= kwargs.get('alpha', 0.75)
    linewidths = kwargs.get('linewidths', 1)
    edgecolors = kwargs.get('edgecolors', None)
    facecolors = kwargs.get('facecolors', None)
    directioncolor = kwargs.get('directioncolor', 'r')
    # edge properties
    connectionstyle = kwargs.get('connectionstyle', None)
    
    # node/edge labels properties
    font_size = kwargs.get('font_size', font_size)
    
    fig = None
    
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    try:
        g = deepcopy(g)
    except Exception as e:
        print(e)

    g = g.to('cpu')
    g.ndata['_N'] = g.nodes()
    
    if extent is not None:
        x_min, x_max = extent['x_min']-pad[0], extent['x_max']+pad[1]
        y_min, y_max = extent['y_min']-pad[2], extent['y_max']+pad[3]
        # x_min, y_min, x_max, y_max = extent

    else:
        x_min, y_min = g.ndata['pos'].min(0).values.cpu().numpy()
        x_max, y_max = g.ndata['pos'].max(0).values.cpu().numpy()
        
    width = x_max - x_min
    height = y_max - y_min
    aspect_ratio = width/height    
    x_margin = np.minimum(width/2, min_margin)
    y_margin = np.minimum(height/2, min_margin)    
    
    if ax is None:
        plt.close('all')
        fig = plt.figure(fig_name, figsize=figsize, dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        # ax = fig.add_axes([0, 0, 1, 1 / aspect_ratio]) # aspect ration not working for some cases
        ax.axes.set_aspect('equal', 'datalim')

    if axis_off:
        ax.axis('off')
    
    if limit_axes:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax.set_title(fig_name)
    
    if frame is not None:
        if extent is not None:
            ax.imshow(frame, aspect='auto', extent=[x_min, x_max, y_min, y_max]) #extents = (left, right, bottom, top)
        else:
            print('Showing frame without extent, may not render properly. Provide x_min, x_max, y_min, y_max to define the extent of the frame')
            ax.imshow(frame, aspect='auto')

    node_color_label = 'nid'
    # unique_ped = g.ndata['tid'].unique().numpy()
    unique_ped = g.ndata[node_color_label].unique().numpy()
    ped_colors = np.random.random((3, len(unique_ped)))
    
    # node
    pos = g.ndata[pos_attr].numpy()
    node_labels = g.ndata[node_label].flatten().numpy().astype(int)
    
    node_colors = []
    for u in g.nodes():
        
        nid = g.ndata[node_color_label][u].item()
        # color = line_colors[nid % len(line_colors)]
        color = get_color(nid, line_colors)
    
        if g.ndata['cid'][u]==2:
            color = '#808080'
            
        node_colors.append(color)
        
    # draw nodes
    node_collection=ax.scatter(pos[:,0], pos[:,1],
                               s=node_size,
                               c=node_colors,
                               marker=node_shape,
                               alpha=0.5,
                               linewidths=linewidths,
                               edgecolors=edgecolors,
                               facecolors=facecolors,
                               zorder=10
                               )

    node_collection=ax.scatter(pos[:,0], pos[:,1],
                               s=10,
                               c='k',
                               marker=node_shape,
                               alpha=0.9,
                               linewidths=linewidths,
                               edgecolors=edgecolors,
                               facecolors=facecolors,
                               zorder=10
                               )

    
    # draw node labels
    if show_node_label:
        # node_labels = g.ndata[node_label].numpy()
        draw_node_labels(pos, node_labels, font_size=font_size, ax=ax, zorder=10)
    
    # # draw edges
    src_nodes, dst_nodes = g.edges()
    edges = [(u,  v) for u, v in zip(src_nodes.numpy(), dst_nodes.numpy())]
    
    temporal_edges = np.array(edges)[g.edata['spatial_mask'].flatten()==0]
    spatial_edges = np.array(edges)[g.edata['spatial_mask'].flatten()==1]
    spatial_edge_width = [min(1, 1/g.edata['dist'][g.edge_ids(u, v)].numpy()) for u, v in spatial_edges]
    
    draw_edges(pos, temporal_edges, node_size=node_size, ax=ax, alpha=0.8, zorder=9)
    draw_edges(pos, spatial_edges, width=spatial_edge_width, node_size=node_size, connectionstyle=connectionstyle, 
                ax=ax, alpha=0.8, zorder=9)
    
    if show_edge_label:
        edge_labels = {(u, v):g.edata[edge_label][g.edge_ids(u, v)].numpy().round(2) for u, v in spatial_edges}
        edge_labels = {k: tuple(v) if len(v)>1 else v.item() for k, v in edge_labels.items()}
        draw_edge_labels(pos, edge_labels, font_size=font_size, alpha=alpha, ax=ax, zorder=6)

    if show_legend:
        # custom legends
        legend_elements = []
        for p in unique_ped:
            # color = get_color(tid)
            # color = line_colors[tid % len(line_colors)]
            color = get_color(p, line_colors)
            legend_elements.append(Line2D([0], [0], marker='o', color=color, markerfacecolor=color, markersize=4,
                                            # label='{}:{}'.format(tid, g.nodes()[g.ndata['tid']==tid].tolist()),
                                           label='ped-{}'.format(p) # indexed to 0
                                           # label='tid-{}, cid-{}'.format(tid, g.ndata['cid'][g.ndata['tid']==tid][0].item())
                                           ))
        ax.legend(handles=legend_elements, fontsize='small', handlelength=1)

    if show_direction:
        
        current_pos = g.ndata['pos'].cpu().numpy()
        current_vel = g.ndata['vel'].cpu().numpy()
        # current_vel = g.ndata['dir'].cpu().numpy()
        for i in range(len(current_vel)):
            
            x, y = current_pos[i]
            dx, dy = current_vel[i]
            dnorm = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
            if dnorm>0:
                dx = dx/dnorm
                dy = dy/dnorm
            ax.quiver(x, y, dx, dy, color=directioncolor, zorder=10,
                      # units='xy', 
                       scale=4, scale_units='xy', angles='xy',
                       width=0.003,
                      # width=0.005*(y_max-y_min),
                      # width=0.01*(y-y_min)/(y_max-y_min),
                      headwidth=3, headlength=3, headaxislength=3,
                      )

    if dtext is not None:
        # plt.text(0.05, 0.96, '%s, frame:%s'%(dtext, counter), transform=ax.transAxes, fontsize=16, color='blue', va='top',)
        plt.text(0.01, 0.96, '%s'%dtext, transform=ax.transAxes, fontsize=16, color='blue', va='top', weight='bold')
        
    if save_dir is not None:
        if fprefix is not None:
            file_path = save_dir + '/{}_frame_{}.jpeg'.format(fprefix, counter)
            plt.title(fprefix)
        else:
            file_path = save_dir + '/frame_{}.jpeg'.format(counter)
        plt.savefig(file_path , bbox_inches='tight',dpi=100)

    return fig, ax
        
def draw_node_labels(pos, labels, font_size=12, font_color='k', font_family="sans-serif", font_weight="normal", 
                     horizontalalignment="center", verticalalignment="center",
                     alpha=None, bbox=None, ax=None, zorder=5):

    if ax is None:
        ax = plt.gca()
    
    if labels is None:
        return None
    
    text_items={}  # there is no text collection so we'll fake one
    for n, label in enumerate(labels):
        (x, y) = pos[n]
        label=str(label) # this will cause "1" and 1 to be labeled the same
        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            transform=ax.transData,
            bbox=bbox,
            clip_on=True,
            zorder=zorder,
            
            )
        text_items[n] = t
        
    return text_items

def draw_edges(pos, edgelist, width=1.0, edge_color='k', linestyle='solid', arrowstyle='-|>',
               node_size=100, alpha=1., label=None, directed=True, arrow_scale=10, 
               connectionstyle=None, ax=None, zorder=1, **kwargs):

    if ax is None:
        ax=pylab.gca()

    if len(edgelist)==0: # no edges!
        return None

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]],pos[e[1]]) for e in edgelist])

    arrow_collection = []
    # FancyArrowPatch doesn't handle color strings
    arrow_color = colorConverter.to_rgba_array(edge_color, alpha)[0]
    for i, (src, dst) in enumerate(edge_pos):
        x1, y1 = src
        x2, y2 = dst

        if np.iterable(width):
            if len(width) == len(edge_pos):
                line_width = width[i]
            else:
                line_width = width[i % len(width)]
        else:
            line_width = width
            
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=arrowstyle,
            # shrinkA=shrink_source,
            # shrinkB=shrink_target,
            mutation_scale=arrow_scale,
            color=arrow_color,
            linewidth=line_width,
            connectionstyle=connectionstyle,
            linestyle=linestyle,
            zorder=zorder,
        )  # arrows go behind nodes

        # There seems to be a bug in matplotlib to make collections of
        # FancyArrowPatch instances. Until fixed, the patches are added
        # individually to the axes instance.
        arrow_collection.append(arrow)
        ax.add_patch(arrow)
            
    return arrow_collection

def draw_edge_labels(pos, labels, label_pos=0.3, font_size=10, alpha=None, ax=None, rotate=True, 
                     font_color='k', font_family="sans-serif", font_weight="normal", bbox=None,
                         horizontalalignment="left", verticalalignment="center", zorder=1):
    if ax is None:
        ax = plt.gca()

    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        # (x, y) = (
        #     x1 * label_pos + x2 * (1.0 - label_pos),
        #     y1 * label_pos + y2 * (1.0 - label_pos),
        # )

        # label position are at the end of arrow
        (x, y) = (
            x2 - label_pos * (x2-x1), 
            y2 - label_pos * (y2-y1)
            )

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=zorder,
            clip_on=True,
        )
        text_items[(n1, n2)] = t
        
    return text_items


if __name__=='__main__':
    import torch
    show_node_label=False
    show_edge_label=False
    node_label='nid'
    edge_label='id'
    pos_attr='pos'
    edge_attr='dist'
    node_size=100
    rad=0.04
    show_legend=False
    save_dir=None
    fprefix=None
    frame=None
    counter=0
    pad=(0, 0, 0, 0)
    extent=None
    
    # g = raw_data['obsv_graphs']
    # sigma_e = torch.sigmoid(g.edata['e']).mean(-1)
    # g.edata['sigma_e'] = sigma_e
    plt.close('all')
    fig, ax = network_draw(g, show_node_label, node_label, node_size=node_size, linewidths=1, edgecolors='k')
    network_draw(g1, show_node_label, node_label, pos_attr='pred_pos', ax=ax, extent=patch_box, node_size=node_size, linewidths=1, )