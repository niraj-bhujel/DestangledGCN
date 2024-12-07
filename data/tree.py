#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:12:24 2022

@author: dl-asoro
"""
from copy import copy
class Tree():
    def __init__(self, root):
        self.root = root
        self.children = []
        
    def addNode(self,obj):
        self.children.append(obj)
        
        
    def get_paths(self, paths=None, current_path=None):
        if paths is None:
            paths = []
        if current_path is None:
            current_path = []
    
        current_path.append(self.root)
        if len(self.children) == 0:
            paths.append(current_path)
        else:
            for child in self.children:
                self.get_paths(child, paths, current_path)
        return paths

class Node():
    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent
        self.children = []
        
    def addNode(self, obj):
        self.children.append(obj)


def get_all_paths(root, paths=None, current_path=None):
     
    if paths is None:
        paths = []
    if current_path is None:
        current_path = []
     
    # If root is null
    if (not root):
        return
 
    # Insert current node's data into the vector
    current_path.append(root.data)
 
    # If current node is a leaf node
    if (len(root.children) == 0):
 
        paths.append(copy(current_path))
        
        # Pop the leaf node and return
        current_path.pop()
        return
 
    # Recur for all children of the current node
    for child in root.children:
 
        # Recursive Function Call
        get_all_paths(child, paths, current_path)
         
    current_path.pop()
    
    return paths


if __name__ == '__main__':
    paths = get_all_paths(lane_tree)
    