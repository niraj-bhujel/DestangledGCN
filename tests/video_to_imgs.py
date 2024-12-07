#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:46:52 2023

@author: dl-asoro
"""
import os
import cv2
import os.path as osp
import numpy as np
import argparse
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
def _cv2_rotate(im, orientation=0):
    # Rotate a cv2 video manually
    if orientation == 0:
        return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 180:
        return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 90:
        return cv2.rotate(im, cv2.ROTATE_180)
    return im
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default= ROOT +'/video.mp4', help='Path to video.')
    parser.add_argument('--format', type=str, default='jpg', help='Output image format')
    parser.add_argument('--fps', type=float, default=15, help='Output FPS')
    parser.add_argument('--save_dir', type=str, default=ROOT + '/video/', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--view-img', action='store_true', help='show results')
    args = parser.parse_args()
    return args
    
if __name__=='__main__':
    
    args = parse_args()

    # args.source = '/home/dl-asoro/Downloads/Level9_South_Demo.mp4'
    
    if not osp.isfile(args.source):
        raise Exception("%s doesn't exists!!"%args.source)
        
    # args.save_dir = osp.join(osp.dirname(args.source), osp.basename(args.source).split('.')[0])
    
    if osp.exists(args.save_dir):
        shutil.rmtree(args.save_dir, ignore_errors=True)
        
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    
    cap = cv2.VideoCapture(args.source)
    
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print('Video found! Frames: %d, Duration: %d, FPS:%d'%(frames, duration, fps))
        
    if args.fps> fps:
        args.fps = fps
        print('Output frame rate (%d FPS) cannot be greater than original frame rate (%d FPS). Setting output FPS to %d'%(args.fps, fps, fps))
    
    fps_ratio = int(fps/args.fps)
    
    success, img = cap.read()
    
    count = 0
    frame = 0
    
    while success:
        count += 1
        
        frame_duration = count / fps
                
        if count%fps_ratio==0:
            frame += 1
            img = _cv2_rotate(img, orientation=0)
            cv2.imwrite(args.save_dir + '/' + '%d'%frame + '.'  + args.format, img)
            print('Image %d extracted to %s'%(frame, args.save_dir))
        
        success, img = cap.read()
    
    # print(frame, 'frames processed to', args.save_dir)
        