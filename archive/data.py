# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:27 2022, @author: Simon, @version: 1.0
"""
from misc import encode

import json
from pathlib import Path
import numpy as np
import h5py

class Data(dict):
    """
    A general class to handle and store data, working as an upgraded dictionary.
    An empty container can be created via d=Data(), and filled by setting d['key']=value.
    Or an existing json or HDF file can be loaded via d=Data(path).
    The hierarchical tree structure can be maintained; or every key can be aligned to the root object.
    All keys are also accessible as attributes of the object, enabling auto-completion.
    """
    # getter_exceptions = ['_ipython_canary_method_should_not_exist_',
    #                      '_ipython_display_','_repr_mimebundle_','__wrapped__']
    
    def __init__(self,arg=None,select=None):
        super().__init__(self)
        if arg is not None and isinstance(arg,str):
            self.update(self.load(arg),select=select)
        elif arg is not None:
            self.update(arg,select=select)
            
    def __setitem__(self,k,v):
        setattr(self,k,v)
        
    def __setattr__(self,k,v):
        if isinstance(v,dict): 
            v = Data(v)
        self.__dict__[k] = v
        super().__setitem__(k,v)
                
    def __str__(self,rank=0):
        s = '\n'
        for k,v in self.items():
            s += ''.join(['> ' for i in range(rank)]) + str(k) + ':\t'
            if isinstance(v,dict):
                s += v.__str__(rank=rank+1)
            else:
                str_v = str(v)
                if len(str_v)>500:
                    str_v = '\n '.join(str_v.split('\n'))
                    str_v = str_v[:200] + '\n ...\n ' + str_v[-200:]
                s += str_v + '\n'
        return s
    
    def get(self,k,default=None):
        _ = self
        for sub_k in k.split('.'):
            if not hasattr(_,sub_k):
                return default
            _ = getattr(_,sub_k)
        return _
        
    def update(self,arg,select=None):#*args,**kwargs):
        if type(arg) in [h5py._hl.files.File,h5py._hl.group.Group,
                         h5py._hl.dataset.Dataset]:
            self.update(dict(arg.attrs))
        # D = dict(*args,**kwargs)
        # D = {'_'.join(k.split(' ')):v for k,v in dict(arg).items()}
        # print(arg)
        for k,v in dict(arg).items(): 
            k = '_'.join(k.split(' '))
            if select is None or k in select:
                if isinstance(v,dict):
                    self[k] = Data(v)
                # elif type(D[k]) is h5py._hl.files.File:
                    # self[k] = Data(dict(D[k]))
                    # self[k].update(D[k].attrs)
                elif type(v) is h5py._hl.group.Group:
                    self[k] = Data(v)
                    # self[k].update(D[k].attrs)
                elif type(v) is h5py._hl.dataset.Dataset:
                    self[k] = np.array(v)
                    # self.update(D[k].attrs)
                else:
                    self[k] = v
    
    def load(self,path):
        try:
            if path.split('.')[-1]=='json':
                with open(path) as f: 
                    return json.load(f)
            elif path.split('.')[-1]=='h5':
                return h5py.File(path,'r')
            elif path.split('.')[-1] in ['dat','txt']:
                header = open(path).readline()[1:-1].split('\t')
                array = np.loadtxt(path)
                return {k:array[:,i] for i,k in enumerate(header) if k!=''} 
            elif path.split('.')[-1]=='csv':
                return np.genfromtxt(path,skip_header=True)
                # with open(path,'r') as f:
                #     return np.array(list(csv.reader(f,delimiter=';')))
            else: raise Exception('Unknown file format')
        except Exception: print(path)
    
    def save(self,path):
        for sep in ['/','\\']:
            if len(path.split(sep))>1:
                folder = sep.join(path.split(sep)[:-1])
                Path(folder).mkdir(parents=True,exist_ok=True)
        D = {k:encode(v) for k,v in self.items() if k[:2]!='__'}
        with open(path,'w') as f: json.dump(D,f)     
