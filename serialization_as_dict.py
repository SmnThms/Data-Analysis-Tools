# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:51:20 2023, @author: Simon
"""

import numpy as np
import h5py
from fractions import Fraction
from decimal import Decimal
from mpmath import mpf,mpc
import copy

# time, datetime, etc. objects ?

def to_basic_types(dictionary):
    basic_dict = {}
    for k,v in dictionary.items():
        if isinstance(v,dict):
            basic_dict[k] = to_basic_types(v)
        elif isinstance(v,(set,tuple)):
            basic_dict[k] = list(v)
        elif isinstance(v,np.ndarray):
            basic_dict[k] = v.tolist()
        elif isinstance(v,np.integer):
            basic_dict[k] = int(v)
        elif isinstance(v,np.floating):
            basic_dict[k] = float(v)
        elif isinstance(v,(Fraction,Decimal,complex,mpf)):
            basic_dict[k] = str(v)
        elif isinstance(v,mpc):
            basic_dict[k] = str(v).replace(' ','')
        elif isinstance(v,(bool,str,int,float,type(None),list)):
            basic_dict[k] = copy.deepcopy(v)
        else: raise TypeError(f'Encoding of {k} of type {type(v)} not implemented')
    return basic_dict

def h5_to_dict(h5_obj,align_attrs=False):
    if type(h5_obj) is h5py._hl.Dataset.Dataset:
        return {'data':h5_obj[:],'shape':h5_obj.shape,'dtype':h5_obj.dtype,
               'attrs':h5_obj.attrs}
    elif type(h5_obj) in [h5py._hl.files.File,h5py._hl.group.Group]:
        if align_attrs and set(h5_obj.keys())&set(h5_obj.attrs)==set():
            return {**{k:h5_to_dict(v) for k,v in h5_obj.items()},
                    **h5_obj.attrs}