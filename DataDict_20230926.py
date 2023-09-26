# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:27 2022, @author: Simon, @version: 1.0
"""

import numpy as np
import copy
import pandas as pd
import json
import h5py
from fractions import Fraction
from decimal import Decimal
from mpmath import mpf,mpc

convertible_types = (dict,pd.Series,pd.DataFrame,h5py._hl.files.File,
                     h5py._hl.group.Group,h5py._hl.dataset.Dataset)

class DataDict(dict):
    """
A subclass of dict, with convenient tools to navigate through nested dictionaries, 
and conversion option between dict and other hierarchical data formats.

INDENTED STRING REPRESENTATION (with d = DataDict(a_nested_dictionary))
    >>> print(d)
    >>> d.print_keys() # to show the keys only
    >>> d.print_types() # to show the type and shape of the values

PARALLEL ACCESS TO THE KEYS AS ATTRIBUTES (enabling autocompletion in iPython)
    >>> d['key1']['key1a']
    >>> d.key1.key1a        # only for keys that are of type str
    >>> d.get('key1.key1a') # "keypath" (dot-separated chain of keys) accepted

CONVERSION TO A DICT, recursively, of several kinds of objects (pd.Series, 
h5py.File…) and files (.json, .h5…)
    >>> d = DataDict(object)
    >>> d = DataDict(filepath.extension)

CONVERSION FROM A DICT to several kinds of objects and files
    >>> obj = d.to_json_serializable() / …
    >>> d.save(filepath.extension)

EXTENSION OF THE DICT METHODS
    .merge(other_dict): variant of the update method, that works recursively
         for nested dicts (and prevents overwriting in case of key collision).
         
    .rename({old_key:new_key}): returns a copy of the dict, with new_keys 
         being substituted to old_keys.
         
	.select(list_of_keys): returns the corresponding portion of the dictionary 
             (here as well, "keypaths" are supported as keys).
             
	.search(key): returns a list of all "keypaths" leading to the searched key.
    
	.get_above/below(test_key): returns the value correponding to the closest 
             key that is above or below the test_key (in alphabetical order).

N.B.: the pop method also supports a "keypath" argument; 
      the copy method returns a deep copy of the dict.
    """
    
    def __init__(self,arg=None):
        super().__init__(self)
        if isinstance(arg,str):
            arg = self.__load__(arg)
        if arg is not None:
            for k,v in DataDict.__convert__(arg).items():
                if isinstance(v,convertible_types):
                    self[k] = DataDict(v)
                else:
                    self[k] = v
            
    def __setitem__(self,k,v):
        self.__setattr__(k,v)
            
    def __setattr__(self,k,v):
        if type(v) is dict: 
            v = DataDict(v)
        super().__setitem__(k,v)
        if isinstance(k,str):
            self.__dict__[k] = v
        
    def __indented_repr__(self,display_function,indent='> '):
        res = ''
        for k,v in self.items():
            header = indent + str(k) + ': '
            res += header
            if isinstance(v,DataDict):
                res += '\n' + v.__indented_repr__(display_function,indent+'> ')
            else:
                content = str(display_function(v))
                content = content.replace('\n','\n'+' '*len(header))
                if len(content)>500:
                    content = content[:200] + '\n...\n' + content[-200:]
                res += content + '\n'
        return res
                
    def __str__(self):
        return self.__indented_repr__(lambda v:v)
        
    def __repr__(self):
        return self.__str__()
    
    def print_keys(self):
        print(self.__indented_repr__(lambda v:''))
        
    def print_types(self):
        def type_and_shape(v):
            res = type(v).__name__
            if isinstance(v,(list,tuple,set,np.ndarray)):
                res += ' ' + str(np.shape(v))
            return res
        print(self.__indented_repr__(type_and_shape))
        
    def update(self,other):
        super().update(DataDict(other))
                                    
    def merge(self,other,no_overwriting=True):
        if type(other) is not DataDict:
            other = DataDict(other)
        for k in set(other)&set(self):
            if isinstance(self[k],dict) and isinstance(other[k],dict):
                self[k].merge(other[k],no_overwriting)
            elif no_overwriting and self[k]!=other[k]:
                try: raise Exception('Attempt to overwrite '+k)
                except Exception: raise
            else:
                self[k] = other[k]
        for k in set(other)-set(self):
            self[k] = other[k]
            
    def pop(self,key,separator='.'):
        if separator not in key:
            super().pop(key)
        else:
            branch,leaf = key.rsplit(separator,maxsplit=1)
            parent = self.get(branch)
            parent.pop(leaf)
            if parent=={}:
                self.pop(parent,separator)
    
    def get(self,key,default=None,separator='.'):
        if key in self:
            return self[key]
        if separator in key: 
            root,branch = key.split(separator,maxsplit=1)
            if isinstance(self.get(root),dict):
                return self.get(root).get(branch,default,separator)
        return default
    
    def get_below(self,key):
        return self.get(max([k for k in self.keys() if k<=key]))
    
    def get_above(self,key):
        return self.get(min([k for k in self.keys() if k>=key]))
        
    def select(self,keys,separator='.'):
        d = DataDict()
        for key in keys:
            d.merge(DataDict.__expand_keypath__(key,self.get(key),separator))
        return d
    
    def search(self,key,partial=False,root='',separator='.'):
        res = []
        if root!='': root += separator
        for k in self:
            if (not partial and key==k) or (partial and key in k):
                res.append(root+k)
            if isinstance(self[k],DataDict): 
                res += self[k].search(key,root+k,separator)
        return res
    
    def rename(self,conversion_dict):
        for old_key,new_key in conversion_dict.items(): 
            if old_key in self:
                self[new_key] = self[old_key]
                self.pop(old_key)
                
    def copy(self):
        return copy.deepcopy(self)
    
    @staticmethod
    def __expand_keypath__(key,value,separator='.'):
        if isinstance(key,str) and separator in key:
            branch,leaf = key.rsplit(separator,maxsplit=1)
            return DataDict.__expand_keypath__(branch,{leaf:value})
        else:
            return {key:value}
    
    @staticmethod
    def __convert__(obj):
        if isinstance(obj,pd.Series):
            return dict(obj)
        if isinstance(obj,pd.DataFrame):
            return obj.to_dict(orient='index')
        if isinstance(obj,(h5py._hl.files.File,
                           h5py._hl.group.Group,
                           h5py._hl.dataset.Dataset)):
            attrs = dict(obj.attrs)
            attrs_dict = {'attrs':attrs} if len(attrs)!=0 else {}
            if isinstance(obj,h5py._hl.dataset.Dataset):
                if obj.shape!=():
                    content = {'data':obj[:],'shape':obj.shape,'dtype':obj.dtype}
                else: content = {}
            else:
                content = dict(obj)
            return {**attrs_dict,**content}
        else:
            return obj
    
    @staticmethod
    def __encode__(x,name=None):
        if isinstance(x,(bool,str,int,float,type(None),list,dict)):
            return x # copy.deepcopy(x)
        elif isinstance(x,(set,tuple)):
            return list(x)
        elif isinstance(x,type):
            return str(x.__name__)
        elif isinstance(x,np.dtype):
            return str(x.name)
        elif isinstance(x,np.ndarray):
            return x.tolist()
        elif isinstance(x,np.integer):
            return int(x)
        elif isinstance(x,np.floating):
            return float(x)
        elif isinstance(x,(Fraction,Decimal,complex,mpf)):
            return str(x)
        elif isinstance(x,mpc):
            return str(x).replace(' ','')
        else: 
            warning = f'Encoding of type {type(x)} not implemented'
            if name is not None: warning += f' ({name})'
            raise TypeError(warning)
            
    def to_json_serializable(self):
        return {k:DataDict.__encode__(v,k) for k,v in self.items()}
        
    def __load__(self,path):
        extension = path.rsplit('.',1)[1]
        if extension=='json':
            with open(path) as f:
                return json.load(f)
        elif extension=='h5':
            self.h5_file = h5py.File(path,'r')
            return self.h5_file
        # elif extension in ['dat','txt']:
        #     header = open(path).readline()[1:-1].split('\t')
        #     array = np.loadtxt(path)
        #     return {k:array[:,i] for i,k in enumerate(header) if k!=''} 
        # elif extension=='csv':
            # return np.genfromtxt(path,skip_header=True)
            # with open(path,'r') as f:
                # return np.array(list(csv.reader(f,delimiter=';')))
        else: raise Exception('Unknown file extension')
        
    def save(self,path):
        # folder,_ = os.path.split(path)
        # Path(folder).mkdir(parents=True,exist_ok=True)
        extension = path.rsplit('.',1)[1]
        if extension=='json':
            with open(path,'w') as f:
                json.dump(self.to_json_serializable(),f)
        
    
d = {'a': 3.141592653589793,
  'b': True,
  'c': {'c1': np.array([[2.3, 2.3],
          [2.3, 2.3]]), 'c2': {'c21': None, 'c22': np.inf,'cjhekjhlfkqjezhl':'C:/Users/Simon/Documents/0_BEC/1_DataDict_TREATMENT/dada/handle/DataDict2.py','ttttker fjior':np.diag([1,2,3,4,5])}},
  'd': 'Bonjour',
  'e': Fraction(1, 7),
  'fgjhre fhjiheri djh':np.diag(np.ones(123)),
  'oir':[1,2,3,4]*46,
  'z':[[1,2],[3,4]],
  8:1}
D = DataDict(d)
# # print(D)
# D.print_types()
D.merge({'c':{'c2':{'c23':0}},9:3})
# print(D)
e = D.c.copy()
g = D.select(['a','b','c.c1'])
e.update(g)

# S1 = pd.Series({'a1':0,'a2':1})
# S = pd.Series({'a':S1,'b':np.array([[1,2],[3,4]])})
# s = DataDict(S)

# f = h5py.File('2023-06-19_0003_seqBEC_0.h5')
# F = DataDict(f)
# o = F.globals.Dimple
# o.update(S)

u = {'col1': [0, 1, 2, 3], 'col2': pd.Series([2, 3], index=[2, 3])}
v = pd.DataFrame(data=u, index=[0, 1, 2, 3])
w = DataDict(v)