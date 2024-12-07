#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:16:21 2020

@author: dl-asoro
"""
import os
import sys
import time
import shutil
import math
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import dgl
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

data_min_max = {'eth': {'x_min': -7.69, 'x_max': 14.42, 'y_min': -3.17, 'y_max': 13.21},
              'hotel': {'x_min': -3.25, 'x_max': 4.35, 'y_min': -10.31, 'y_max': 4.31},
              'univ': {'x_min': -0.462, 'x_max': 15.469, 'y_min': -0.318, 'y_max': 13.892},
              'zara1': {'x_min': -0.14, 'x_max': 15.481, 'y_min': -0.375, 'y_max': 12.386},
              'zara2': {'x_min': -0.358, 'x_max': 15.558, 'y_min': -0.274, 'y_max': 13.943}
              } 

data_stats = {'eth': {'x_center': 5.264,
              'y_center': 5.315,
              'x_scale': 4.993,
              'y_scale': 1.897,
              'x_min': -7.69,
              'y_min': -3.17,
              'x_max': 14.42,
              'y_max': 13.21},
             'hotel': {'x_center': 1.394,
              'y_center': -2.979,
              'x_scale': 1.586,
              'y_scale': 3.901,
              'x_min': -3.25,
              'y_min': -10.31,
              'x_max': 4.35,
              'y_max': 4.31},
             'univ': {'x_center': 8.299,
              'y_center': 7.302,
              'x_scale': 4.163,
              'y_scale': 3.225,
              'x_min': -0.462,
              'y_min': -0.318,
              'x_max': 15.469,
              'y_max': 13.892},
             'zara1': {'x_center': 7.09,
              'y_center': 4.897,
              'x_scale': 4.515,
              'y_scale': 1.554,
              'x_min': -0.14,
              'y_min': -0.375,
              'x_max': 15.481,
              'y_max': 12.386},
             'zara2': {'x_center': 6.626,
              'y_center': 5.961,
              'x_scale': 4.117,
              'y_scale': 1.585,
              'x_min': -0.358,
              'y_min': -0.274,
              'x_max': 15.558,
              'y_max': 13.943}
              }
# line_colors = ['#ff0000', '#f47d33', '#10b798', '#9d0208', '#002e4f',  '#ff00be', '#54b948',]
line_colors = ['#ff0000', '#10b798', '#dc417f', '#172774', '#ff0080', '#f47d33', '#9772FB', '#9d0208', '#04724d', '#6FDFDF']

def get_color(idx, colors=None):
    if colors is not None:
        return colors[idx%len(colors)]
    idx = idx * 3
    color = [(37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255]
    color = [c/255 for c in color]
    return color

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """

    color = line.get_color()
    
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    # arrow between the positions xy and xytext
    line.axes.annotate('',
                       xy=(xdata[-1], ydata[-1]),
                       xytext=(xdata[-2], ydata[-2]),
                       arrowprops=dict(arrowstyle='simple',  color=color),
                       size=size*10
                       )
    
def plot_path(obsv_traj, trgt_traj, pred_traj=None, ped_ids=None, K=1, extent=None, pad=(1, 1, 1, 1), 
              counter=0, frame=None, save_dir=None, dtext='', fprefix=None, legend=False, ax=None, figsize=(8, 5), 
              axis_off=True, limit_axes=False, arrow=False, ticks_off=False, min_obsv_len=2, min_pred_len=2, max_pred_len=12,
              lw=2, lm='o', ms=1, mw=1, dpi=100):
    '''
    Parameters
    ----------
    obsv_traj : List of N arrays each with shape [ped_obsv_len, 2]
    trgt_traj : List of N arrays each with shape [ped_trgt_len, 2]
    pred_traj : List of N arrays each with shape [K, ped_trgt_len, 2]
    ped_ids: List of N id
    K: number of prediction to plot
    counter : TYPE, optional
        DESCRIPTION. The default is 0.
    frame : TYPE, optional
        DESCRIPTION. The default is None.
    save_dir : TYPE, optional
        DESCRIPTION. The default is './plots'.
    legend : TYPE, optional
        DESCRIPTION. The default is False.
    axis_off : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    if isinstance(obsv_traj[0], torch.Tensor):
        obsv_traj = [t.cpu().numpy() for t in obsv_traj]
    if isinstance(trgt_traj[0], torch.Tensor):
        trgt_traj = [t.cpu().numpy() for t in trgt_traj]
    if pred_traj is not None:
        if isinstance(pred_traj[0], torch.Tensor):
            pred_traj = [t.cpu().numpy() for t in pred_traj]
        
        K = min(K, pred_traj[0].shape[0])

    if extent is not None:
        x_min, x_max = extent['x_min']-pad[0], extent['x_max']+pad[1]
        y_min, y_max = extent['y_min']-pad[2], extent['y_max']+pad[3]
        # x_min, x_max, y_min, y_max = extent['x_min'], extent['x_max'], extent['y_min'], extent['y_max']
    else:
        seq_traj = np.concatenate(obsv_traj + trgt_traj)
        x_min, y_min = seq_traj.min(axis=0) - pad[:2]
        x_max, y_max = seq_traj.max(axis=0) + pad[2:]

    x_min = min(-0.5, x_min)
    y_min = min(-0.5, y_min)
    x_max = max(0.5, x_max)
    y_max = max(0.5, y_max)

    #create canvass
    plt.close('all')
    if ax is None:
        fig = plt.figure(frameon=True, figsize=figsize)
        ax = plt.axes()
        fig.add_axes(ax)

    if axis_off:
        ax.axis('off')

    if frame is not None:
        if extent is None:
            print('Showing frame without extent, may not render properly. Provide x_min, x_max, y_min, y_max to define the extent of the frame')
            # ax.imshow(frame, aspect='auto')
        ax.imshow(frame, aspect='auto', extent=[x_min, x_max, y_min, y_max]) #extents = (left, right, bottom, top), default is (-0.5, numcols-0.5, numrows-0.5, -0.5)
        
    # set limit, useful when to prevent plotting outside frame
    # if limit_axes:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.plot(0, 0, 'o', color='black')
    # ax.plot((x_max + x_min)/2, (y_max+y_min)/2, 's', color='blue')
    
    num_peds = len(obsv_traj)
    cmap = plt.cm.get_cmap(name='Set1', lut=num_peds)
    # pred_cmap = plt.cm.get_cmap(name='Set3', lut=num_peds)
    if ped_ids is not None:
        assert num_peds == len(ped_ids), 'number of traj not equal to ped_ids'


    legend_handles = []
    legend_labels = []    
    for p in range(num_peds):
        # color = cmap(p)
        if ped_ids is not None:
            color = line_colors[ped_ids[p]%(len(line_colors))]
        else:
            color = line_colors[p%(len(line_colors))]

        # quiver requires at least two steps
        if len(trgt_traj[p])<min_pred_len:
            continue

        if len(obsv_traj[p])<min_obsv_len:
            continue
            
        # obsv tracks
        xs, ys = obsv_traj[p][:, 0], obsv_traj[p][:, 1]
        
        # start markers
        # start_mark = ax.scatter(xs[:1], ys[:1], c=[color], label='Start', marker=lm, edgecolors='k', s=lw**3, zorder=3)
        start_mark = ax.scatter(xs[-1:], ys[-1], c=[color], label='Start', marker=lm, edgecolors='k', s=lw**3, zorder=3)

        kwargs = {'linewidth':lw, 'marker':lm, 'markersize':ms, 'fillstyle':'full', 'mfc':'w', 'mec':color, 'mew':mw}

        # plot obsv tracks Never Walk Alone: Mod­e
        obsv_line, = ax.plot(xs, ys, color=color, label='Obsv', linestyle='solid', zorder=2, **kwargs)
    
        #target tracks
        xs, ys = trgt_traj[p][:max_pred_len, 0], trgt_traj[p][:max_pred_len, 1]
        path_effects = [pe.Stroke(linewidth=lw, foreground='k'), pe.Normal()]
        target_line, = ax.plot(xs, ys, color='w', label='Target', linestyle='solid', zorder=3, path_effects=path_effects, **kwargs)
        
        if arrow:
            # end marker
            ax.quiver(xs[-2], ys[-2], (xs[-1]-xs[-2])+0.001, (ys[-1]-ys[-2])+0.001, color=color, zorder=3, 
                        angles='xy', scale_units='xy', scale=1, width=0.02*(ys[-1]-y_min)/(y_max-y_min),
                        # headwidth=3, headlength=4, headaxislength=3,
                        )
        
        if pred_traj is not None:
            preds = pred_traj[p][:, :len(trgt_traj[p][:max_pred_len]), :]

            # plot top k predicted traj
            for k in range(K):
                xs, ys = preds[k][:, 0], preds[k][:, 1]
                pred_line, = ax.plot(xs, ys, color=color, label='Predictions', linestyle='dashed', zorder=1, **kwargs)
                if arrow:
                    # end arrow
                    ax.quiver(xs[-2], ys[-2], (xs[-1]-xs[-2])+0.001, (ys[-1]-ys[-2])+0.001, color=color, zorder=3,
                              # angles='xy', scale_units='xy', scale=2, width=0.015*(ys[-1]-y_min)/(y_max-y_min),
                              width=0.01*(ys[-1]-y_min)/(y_max-y_min), headwidth=3, headlength=4, headaxislength=3,
                              )
                    
            if ped_ids is not None:
                legend_handles.append(pred_line)
                legend_labels.append('{}:cid{}'.format(p, int(ped_ids[p])))
            
    if legend:
        legend_handles.extend([start_mark, target_line, pred_line])
        legend_labels.extend(['Start', 'GT', 'Pred'])
        
        ax.legend(legend_handles, legend_labels, handlelength=4)
        
    if dtext is not None:
        # plt.text(0.05, 0.96, '%s, frame:%s'%(dtext, counter), transform=ax.transAxes, fontsize=16, color='blue', va='top',)
        plt.text(0.05, 0.96, '%s'%dtext, transform=ax.transAxes, fontsize=16, color='blue', va='top',)
    
    if ticks_off:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    
    plt.tight_layout()
    
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if fprefix is not None:
            file_path = save_dir + '/{}_frame_{}.jpeg'.format(fprefix, counter)
        else:
            file_path = save_dir + '/frame_{}.jpeg'.format(counter)
        # print(file_path)
        plt.savefig(file_path , bbox_inches='tight',dpi=dpi)
    # plt.show()
    # plt.close(fig)
    return fig, ax


def plot_kde(gt_history, gt_futures, pred_futures, ped_ids=None, extent=None, frame=None, pad=(1, 1, 1, 1), fprefix=None, save_dir=None, 
            figsize=(8, 5), counter=0, ax=None, axis_off=True, line_width=2, sns_thresh=0.05, circle_radius=0.3, alpha=0.8, 
            min_obsv_len=2, min_pred_len=2, max_pred_len=12, dpi=100):

    if type(gt_history[0])==torch.Tensor:
        gt_history = [h.cpu().numpy() for h in gt_history]
    if type(gt_futures[0])==torch.Tensor:
        gt_futures = [f.cpu().numpy() for f in gt_futures]
    if type(pred_futures[0])==torch.Tensor:
        pred_futures = [p.cpu().numpy() for p in pred_futures]

    assert len(gt_history)==len(gt_futures)

    if extent is not None:
        x_min, x_max = extent['x_min']-pad[0], extent['x_max']+pad[1]
        y_min, y_max = extent['y_min']-pad[2], extent['y_max']+pad[3]
        # x_min, y_min, x_max, y_max = extent
    else:
        seq_traj = np.concatenate(gt_history + gt_futures)
        x_min, y_min = seq_traj.min(axis=0) - pad[:2]
        x_max, y_max = seq_traj.max(axis=0) + pad[2:]

    x_min = min(-0.5, x_min)
    y_min = min(-0.5, y_min)
    x_max = max(0.5, x_max)
    y_max = max(0.5, y_max)

    plt.close('all')
    if ax is None:
        fig = plt.figure(frameon=True, figsize=figsize)
        ax = plt.axes()
        fig.add_axes(ax)

    if frame is not None:
        ax.imshow(frame, aspect='auto', extent=[x_min, x_max, y_min, y_max]) #extents = (left, right, bottom, top)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if axis_off:
        ax.axis('off')

    for p in range(len(gt_futures)):

        if ped_ids is not None:
            color = line_colors[ped_ids[p]%(len(line_colors))]
        else:
            color = line_colors[p%(len(line_colors))]

        history = gt_history[p]
        future = gt_futures[p][:max_pred_len]
        prediction = pred_futures[p][:, :len(future), :]

        if len(history)<min_obsv_len:
            continue

        if len(future)<min_pred_len:
            continue

        ax.plot(history[:, 0], history[:, 1], '--', color=color, linewidth=line_width, zorder=600)

        for t in range(prediction.shape[1]):
            sns.kdeplot(x=prediction[:, t, 0], y=prediction[:, t, 1], 
                        ax=ax, shade=True, thresh=sns_thresh, warn_singular=False,
                        color=color, alpha=alpha, zorder=650)

        ax.plot(future[:, 0],
                future[:, 1],
                'w--',
                linewidth=line_width-1,
                zorder=700,
                path_effects=[pe.Stroke(linewidth=line_width+3, foreground=color), pe.Normal()]
                )
        
        # Current Node Position
        circle = plt.Circle((history[-1, 0],
                            history[-1, 1]),
                            radius=circle_radius,
                            facecolor=color,
                            edgecolor='k',
                            lw=2,
                            zorder=600)
        ax.add_artist(circle)

    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if fprefix is not None:
            file_path = save_dir + '/{}_frame_{}.jpeg'.format(fprefix, counter)
        else:
            file_path = save_dir + '/frame_{}.jpeg'.format(counter)

        plt.savefig(file_path, bbox_inches='tight',dpi=dpi)

    return fig, ax

          


        