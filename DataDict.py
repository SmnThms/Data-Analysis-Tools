# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:27 2022, @author: Simon, @version: 1.0
"""

import numpy as np
import copy
import json,h5py,csv
from fractions import Fraction

from dict_conversion import from_h5,to_basic_types

class DataDict(dict):
    """
A subclass of dict, with convenient tools to navigate through nested dictionaries, 
and conversion option between dict and other hierarchical data formats.

INDENTED STRING REPRESENTATION: (with d = DataDict(a_nested_dictionary))
    >>> print(d)
    >>> d.print_keys() # to show the keys only
    >>> d.print_types() # to show the type and shape of the values

PARALLEL ACCESS TO THE KEYS AS ATTRIBUTES (enabling autocompletion in iPython):
    >>> d['key1']['key1a'] 
    >>> d.key1.key1a 
    >>> d.get('key1.key1a') # "keypath" (dot-separated chain of keys) accepted

CONVERSION TO A DICT, recursively, of several kinds of objects (pd.Series, 
h5py.File) and files (.json, .h5, .csv):
    >>> d = DataDict(obj)
    >>> d = DataDict(filepath.extension)

CONVERSION FROM A DICT to several kinds of objects (json string, pd.Series, 
h5py.File) and files (.json, .h5, .txt):
    >>> obj = d.to_json() / to_h5File()
    >>> d.save(filepath.extension)

EXTENSION OF THE DICT METHODS:
	.select(list_of_keys): returns the corresponding portion of the dictionary 
             (here as well, "keypaths" are supported as keys).
	.merge(*other_dicts): variant of the update method, recursive when the 
             values are themselves dicts (i.e. handles key conflicts by joining 
             the ).
   	.rename({old_key:new_key}): returns a copy of the dict, with new_keys 
             being substituted to old_keys.
	.search(key): returns a list of all "keypaths" leading to the searched key.
	.get_closest(test_key): returns the value correponding to the closest key 
             of the dict (in numerical or alphabetical order) that is above or 
             below the test_key.

N.B.: the pop method also supports a "keypath" argument; 
      the copy method returns a deep copy of the dict.
    """
    # getter_exceptions = ['_ipython_canary_method_should_not_exist_',
    #                      '_ipython_display_','_repr_mimebundle_','__wrapped__']
    
    def __init__(self,arg=None):
        super().__init__(self)
        if arg is not None and isinstance(arg,str):
            self.update(self.__load__(arg))
        elif arg is not None:
            self.update(arg)
            
    def __setitem__(self,k,v):
        setattr(self,k,v)
            
    def __setattr__(self,k,v):
        if isinstance(v,dict): 
            v = DataDict(v)
        self.__dict__[k] = v
        super().__setitem__(k,v)
    
    def __str__(self,indent='> '):
        s = ''
        for k,v in self.items():
            s += indent + str(k) + ':\t'
            spacing = len(indent+str(k)+':\t')
            if isinstance(v,dict):
                s += '\n' + v.__str__(indent=indent+'> ')
            elif len(str(v))<500:
                s += str(v).replace('\n','\n'+' '*spacing) + '\n'
            else:
                s += str(v)[:200] + '\n[...]\n' + str(v)[-200:] + '\n'
        return s
    
    def __repr__(self):
        return self.__str__()
    
    def print_keys(self,rank=0):
        s = ''
        for k,v in self.items():
            s += ''.join(['> ' for i in range(rank)]) + str(k) + '\n'
            if isinstance(v,dict):
                s += v.print_keys(rank=rank+1)
        if rank==0: print(s)
        else: return s
        
    def print_types(self):
        pass
    
    def copy(self):
        return copy.deepcopy(self)
    
    def get(self,key,default=None,separator='.'):
        if type(key) is list:
            if len(key)==1: key = key[0]
            else: return [self.get(k,default,separator) for k in key]
        if key in self:
            return self[key]
        elif separator in key: 
            head,rest = key.split(separator,maxsplit=1)
            if isinstance(self.get(head),dict):
                return self.get(head).get(rest,default,separator)
        return default
    
    def getlast(self,key):
        return self.get(max([k for k in self.keys() if k<=key]))
    
    def getnext(self,key):
        return self.get(min([k for k in self.keys() if k>=key]))
    
    def search(self,key,root='',separator='.'):
        res = []
        if root!='': root += separator
        for k in self:
            if k==key: 
                res.append(root+k)
            if isinstance(self[k],DataDict): 
                res += self[k].search(key,root+k,separator)
        return res
    
    @staticmethod
    def __expand_key_path__(key,value,separator='.'):
        if separator in key:
            rest,tail = key.rsplit(separator,maxsplit=1)
            return DataDict.expand_key_path(rest,{tail:value})
        else:
            return {key:value}
        
    def pop(self,key,separator='.'):
        if separator not in key:
            super().pop(key)
        else:
            rest,tail = key.rsplit(separator,maxsplit=1)
            node = self.get(rest)
            node.pop(tail)
            if node=={}:
                self.pop(rest,separator)
    
    def select(self,keys,separator='.'):
        d = DataDict()
        for key in keys:
            d = d.merge(DataDict.__expand_key_path__(key,self.get(key),separator),
                        prevent_overwriting=False)
        return d
        
    def update(self,other):
        if not isinstance(other,dict):
            other = dict(other)
        for k,v in other.items(): 
            if isinstance(v,dict):
                self[k] = DataDict(v)
            else:
                self[k] = v
    
    def update_keys(self,conversion_dict):
        for old_key,new_key in conversion_dict.items(): 
            if old_key in self:
                self.update({new_key:self[old_key]})
                self.pop(old_key)
                # value = self.get(old_key.rsplit(separator,1)[0])
                # self.set(self,new_key,value)
                # self.pop(old_key)
                                    
    def merge(self,*others,prevent_overwriting=True): # overriding
        if len(others)>1: 
            return self.merge(others[0]).merge(*others[1:])
        other = others[0]
        if not isinstance(other,dict): 
            other = dict(other)
        for k in set(other) & set(self):
            if isinstance(self[k],dict) and isinstance(other[k],dict):
                self[k] = self[k].merge(other[k],
                                        prevent_overwriting=prevent_overwriting)
            elif prevent_overwriting:
                try: raise Exception('Redundant key: '+k) # Key conflict / collision          
                except Exception: raise
            elif self[k] != other[k]:
                self[k] = other[k]
        for k in set(other) - set(self):
            self[k] = other[k]
        return self
        
    def __load__(self,path):
        extension = path.rsplit('.',1)[1]
        if extension=='json':
            with open(path) as f:
                return json.load(f)
        elif extension=='h5':
            self.h5_file = h5py.File(path,'r')
            return from_h5(self.h5_file)
        elif extension in ['dat','txt']:
            header = open(path).readline()[1:-1].split('\t')
            array = np.loadtxt(path)
            return {k:array[:,i] for i,k in enumerate(header) if k!=''} 
        elif extension=='csv':
            # return np.genfromtxt(path,skip_header=True)
            with open(path,'r') as f:
                return np.array(list(csv.reader(f,delimiter=';')))
        else: raise Exception('Unknown file extension')
        
    def save(self,path):
        # folder,_ = os.path.split(path)
        # Path(folder).mkdir(parents=True,exist_ok=True)
        extension = path.rsplit('.',1)[1]
        if extension=='json':
            with open(path,'w') as f:
                json.dump(self.to_basic(),f)
        # elif extension=='h5':
        #     with open(path,'w')
        
    
d = {'a': 3.141592653589793,
  'b': True,
  'c': {'c1': np.array([[2.3, 2.3],
          [2.3, 2.3]]), 'c2': {'c21': None, 'c22': np.inf,'cjhekjhlfkqjezhl':'C:/Users/Simon/Documents/0_BEC/1_DataDict_TREATMENT/dada/handle/DataDict2.py','fekjhlkejjnfker fjior':np.diag([1,2,3,4,5])}},
  'd': 'Bonjour',
  'e': Fraction(1, 7)}
D = DataDict(d)
# print(DataDict(d).merge({'c':{'c2':{'c23':0}}}))