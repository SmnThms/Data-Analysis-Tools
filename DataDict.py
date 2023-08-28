# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:27 2022, @author: Simon, @version: 1.0
"""

import json
from pathlib import Path
import numpy as np
import os
import h5py
import csv
import copy
from collections.abc import Iterable
import pandas as pd
from fractions import Fraction
from decimal import Decimal
from mpmath import mpf,mpc
from time import time

def u_str(value,uncertainty=None,u_digits=2,
            exponent_separator='x10^',decimal_separator='.'):
        """
        Returns a string in shorthand scientific notation:
            mantissa (last digit uncertainty) x10^ exponent
        
        u_digits: the number of digits in the parenthesis
        exponent_separator: notation introducing the exponent ('x10^','E','e'…)
        """
        exponent = lambda x:int(np.floor(np.log10(abs(x)))) if x!=0 else 0
        separator = lambda s,i:decimal_separator.join([s[:i],s[i:]])        
        if uncertainty is None: uncertainty = 0
        if value!=0:
            exp_v, exp_u = exponent(value), exponent(uncertainty)
            mantissa = lambda x:str(round(x*10**(-min(exp_v,exp_u)+u_digits-1)))
            val_str, unc_str = mantissa(value), mantissa(abs(uncertainty))
            if exp_u<exp_v or u_digits!=1:
                val_str = separator(val_str,{True:1,False:2}[value>0])
            if u_digits!=1:
                if exp_u==exp_v: unc_str = separator(unc_str,1)
                elif exp_u>exp_v: unc_str = separator(unc_str,1-u_digits)
        else:
            val_str, exp_v = '0', exponent(uncertainty)
            unc_str = str(round(abs(uncertainty)*10**(-exp_v+u_digits-1)))
            if u_digits!=1: unc_str = separator(unc_str,1)
        if uncertainty!=0: val_str += '(' + unc_str + ')'
        if exp_v!=0: val_str += exponent_separator + str(exp_v)
        return val_str
    
def measure(dictionary,key,u_digits=2,expsep='E',decsep='.',name=True):
    s = DataDict.shortunc(dictionary[key],dictionary.get('u_'+key),u_digits,expsep,decsep)
    if 'units' in dictionary and isinstance(dictionary['units'],dict): 
        s += ' '+dictionary['units'].get(key,'')
    return key+' = '+s if name else s
    
def arrayify(*args,dtype=None,sort=False,mask=None):
    """
    Converts the arguments as ndarray copies 
    (except for None arguments, wich remain None).
    Applies to all arguments a boolean mask if given.
    If sort==True, sorts all arguments according to the first one.
    """
    res = [np.array(A,dtype=dtype) if A is not None else None for A in args]
    if mask is not None:
        mask = np.asarray(mask,dtype=bool)
        res = [A[mask] if A is not None else None for A in args]
    if sort: 
        order = np.argsort(np.array(args[0]))
        res = [A[order] if A is not None else None for A in args]
    return res if len(res)>1 else res[0]
    
def develop(abbreviation,starting_point=()):
    """
    Examples:
        develop( [(2023,7,5,[2,6])] ) 
            -> [ (2023,7,5,2), (2023,7,5,6) ]
            
        develop( [(2023,7,range(5,8))] )
            -> [ (2023,7,5), (2023,7,6), (2023,7,8)]
            
        develop( [(2022,12,31),(2023,1,1)] ,root=('SeqBEC'))
            -> [ ('SeqBEC',2022,12,31), ('SeqBEC',2023,1,1) ]
            
        develop( [(2023,7,[5, (6,[1,2]) ])] )
            -> [ (2023,7,5), (2023,7,6,1), (2023,7,6,2) ]
    """
    list_of_path_tuples = []
    for path_tuple in abbreviation:
        if type(path_tuple) is not tuple: 
            path_tuple = (path_tuple,)
        if type(path_tuple[-1]) in [list,range,np.ndarray]:
            list_of_path_tuples += develop(path_tuple[-1],
                starting_point = starting_point + path_tuple[:-1])
        else:
            list_of_path_tuples += [starting_point + path_tuple]
    return list_of_path_tuples

def all_paths_ending_with(extension,root='.'):
    paths,n = [],len(extension)
    for branch,_,files in os.walk(root):
        for file in files:
            if file[-n:]==extension:
                paths.append(os.path.join(branch,file))
    return paths

def dict_to_basic(dictionary):
    basic_dict = {}
    for k,v in dictionary.items():
        if isinstance(v,dict):
            basic_dict[k] = to_basic(v)
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
    # return res
    

class DataDict(dict):
    """
    A subclass of dict, with convenient methods to handle nested dictionaries 
and conversion options to other data formats.

In addition to the usual dict methods (get, update, pop, items, keys, values…):
	- merge: returns 
	- select(list of keys): returns the corresponding sub-dictionary
	- copy(): returns a deepcopy of the dict
	- update_keys({old_key:new_key}): substitutes the keys

Navigation options:
	- get, pop and select also support key paths as arguments 
      (with key path of type: separator.join(successive_keys))
	- the keys can also be accessed as attributes (enabling iPython tab completion)
	  ex: d = DataDict({'a':{'b':'hello'}}) -> d['a']['b'] = d.get('a.b') = d.a.b 

Navigation methods:
	- visit(): prints all the keys, recursively in case of a nested dict
	- __str__() (called by print): conveniently indented representation of the dict
	- search(key): returns the path to a given key within the (nested) dict
	- measure(key): returns a string of the value in shorthand scientific notation 
	  (includes the uncertainty if an item 'u_'+key:uncertainty exists, 
	  and the unit if an item 'units':{key:unit} exists)

Conversion options:
	- to_json/from_json: 

Import/export:
	- the __init__ argument can be either an object convertible to a dict, 
	  or the path (including the format extension) to a file convertible to a dict.
	- save(path_including_format_extension): 
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
                s += v.visit(rank=rank+1)
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
    
                    
    # def encode(x):
    #     if isinstance(x,dict): return {k:DataDict.encode(v) for k,v in x.items()}
    #     if isinstance(x,np.ndarray): return x.tolist()
    #     if isinstance(x,np.int32): return int(x)
    #     else: return x
    
    # def encode(obj):
    #     if isinstance(obj, np.integer):
    #         return int(obj)
    #     elif isinstance(obj, np.floating):
    #         return float(obj)
    #         # return np.format_float_scientific(obj,precision=None,trim='-')
    #     elif isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     elif isinstance(obj, (Fraction,Decimal,complex,mpf)):
    #         return str(obj)
    #     elif isinstance(obj,mpc):
    #         return str(obj).replace(' ','')
    #     return
    
    # def to_basic(self):
    #     D = DataDict()
    #     for k,v in self.items():
    #         if isinstance(v,DataDict):
    #             D[k] = v.to_basic()
    #         # elif isinstance(v,dict):
    #         #     D[k] = DataDict(v).to_basic()
    #         elif isinstance(v,(set,tuple)):
    #             D[k] = list(v)
    #         elif isinstance(v,np.ndarray):
    #             D[k] = v.tolist()
    #         elif isinstance(v,np.integer):
    #             D[k] = int(v)
    #         elif isinstance(v,np.floating):
    #             D[k] = float(v)
    #         elif isinstance(v,(Fraction,Decimal,complex,mpf)):
    #             D[k] = str(v)
    #         elif isinstance(v,mpc):
    #             D[k] = str(v).replace(' ','')
    #         elif isinstance(v,(bool,str,int,float,type(None),list)):
    #             D[k] = copy.deepcopy(v)
    #         else: raise TypeError(f'Encoding of {k} of type {type(v)} not implemented')
    #     return D
    
    # def to_json(self):
    #     return json.dumps(self.to_basic())
    
    # def from_json(s):
    #     return DataDict(json.loads(s))
        
    # class Encoder(json.JSONEncoder):
    #     def default(self, obj):
    #         enc = DataDict.encode(obj)
    #         return json.JSONEncoder.default(self,obj) if enc is None else enc
    #     # et encodage en notation scientifique ? (ex. à partir de 1e3)
    #     # avec une résolution relative plancher ?
        
    # def to_h5(self):
    #     pass
    
    # def from_h5(f):
    #     if isinstance(arg,h5py._hl.files.File):
    #         return DataDict.from_h5()
        
    def __load__(self,path):
        extension = path.rsplit('.',1)[1]
        if extension=='json':
            with open(path) as f:
                return json.load(d)
        elif extension=='h5':
            self.h5_file = h5py.File(path,'r')
            return h5_to_dict(self.h5_file)
        elif path.split('.')[-1] in ['dat','txt']:
            header = open(path).readline()[1:-1].split('\t')
            array = np.loadtxt(path)
            return {k:array[:,i] for i,k in enumerate(header) if k!=''} 
        elif path.split('.')[-1]=='csv':
            # return np.genfromtxt(path,skip_header=True)
            with open(path,'r') as f:
                return np.array(list(csv.reader(f,delimiter=';')))
        else: raise Exception('Unknown file extension')
        
    def h5_to_dict(obj,align_attrs=True):
        """récursivement, pour chaque groupe, le convertit en dict et a"""
        res = {}
        if type(obj) in [h5py._hl.files.File,h5py._hl.group.Group]:
            for k,v in obj.items():
                res.update({k:h5_to_dict(v))})
    
        elif type(obj) is h5py._hl.Dataset.Dataset:
        
        if align_attrs:
            d.merge(obj.attrs)
            
                         # ]:
            # self.update(obj.attrs)
            
            
        # if isinstance(obj,str):
        #     return json.dumps(obj)
        # if isinstance(obj,arg,h5py._hl.files.File):
        
    def save(self,path):
        # folder,_ = os.path.split(path)
        # Path(folder).mkdir(parents=True,exist_ok=True)
        extension = path.rsplit('.',1)[1]
        if extension=='json':
            with open(path,'w') as f:
                json.dump(self.to_basic(),f)
        # elif extension=='h5':
        #     with open(path,'w')
        
    # def fromh5(h5):
    #     if type(arg) in [h5py._hl.files.File,h5py._hl.group.Group,
    #                       h5py._hl.Dataset.Dataset]:
    #         self.update(dict(arg.attrs))
    #     # D = dict(*args,**kwargs)
    #     # D = {'_'.join(k.split(' ')):v for k,v in dict(arg).items()}
    #     # print(arg)
    #     for k,v in dict(other).items(): 
    #         k = '_'.join(k.split(' ')) # k.replace(' ','')
    #         if select is None or k in select:
    #         if isinstance(v,dict):
    #             self[k] = DataDict(v)
    #             # elif type(D[k]) is h5py._hl.files.File:
    #                 # self[k] = DataDict(dict(D[k]))
    #                 # self[k].update(D[k].attrs)
    #             elif type(v) is h5py._hl.group.Group:
    #                 self[k] = DataDict(v)
    #                 # self[k].update(D[k].attrs)
    #             elif type(v) is h5py._hl.DataDictset.DataDictset:
    #                 self[k] = np.array(v)
    #     #             # self.update(D[k].attrs)
    
    # def toDatasetAndAttrs(self,N=None):
    #     array,columns,attrs = [],[],[]
    #     if N is None:
    #         N = max([len(v) for v in self.values()])
    #     for k,v in self.items():
    #         if len(v)==N:
    #             array.append(v)
    #             columns.append(k)
    #         else:
    #             attrs.append((k,v))
    
    # def load(self,path):
    #     try:
    #         if path.split('.')[-1]=='json':
    #             with open(path) as f: 
    #                 return json.load(f)
    #         elif path.split('.')[-1]=='h5':
    #             return h5py.File(path,'r')
    #         elif path.split('.')[-1] in ['dat','txt']:
    #             header = open(path).readline()[1:-1].split('\t')
    #             array = np.loadtxt(path)
    #             return {k:array[:,i] for i,k in enumerate(header) if k!=''} 
    #         elif path.split('.')[-1]=='csv':
    #             # return np.genfromtxt(path,skip_header=True)
    #             with open(path,'r') as f:
    #                 return np.array(list(csv.reader(f,delimiter=';')))
    #         else: raise Exception('Unknown file format')
    #     except Exception: print(path)
    
    # def save(self,path):
    #     for sep in ['/','\\']:
    #         if len(path.split(sep))>1:
    #             folder = sep.join(path.split(sep)[:-1])
    #             Path(folder).mkdir(parents=True,exist_ok=True)
    #     D = {k:DataDict.encode(v) for k,v in self.items() if k[:2]!='__'}
    #     with open(path,'w') as f: json.dump(D,f)     

        
# def arrayify(*args,mask=None,sort=False,dtype=None):
#     if mask is None: mask = np.ones(len(args[0]),dtype=bool)
#     if sort: order = np.argsort(np.array(args[0])[mask])
#     else:    order = np.ones(len(np.array(args[0])[mask]),dtype=bool)
#     L = [np.array(A,dtype=dtype)[mask][order] if A is not None 
#          else None for A in args]
#     return L[0] if len(L)==1 else L
#     # if (dtype is sfloat) or (sfloat in [type(x) for x in X]): X
    
# def listify(*args,itemsAreIterable=False):
#     L = []
#     for arg in args:
#         if arg is None: L.append(None)
#         elif not isinstance(arg,Iterable) or type(arg) is str: L.append([arg])
#         elif itemsAreIterable and not isinstance(arg[0],Iterable): L.append([arg])
#         else: L.append(arg)
#     return L

# def isfloat(s):
#     try:
#         float(s)
#         return True
#     except ValueError:
#         return False
    
# def isfloats(L):
#     if len(L)==0: return False
#     return np.all([isfloat(s) for s in L])

# def load_csv(path,delimiter=','):
#     with open(path,'r') as f:
#         rows = csv.reader(f,delimiter=delimiter)
#         A = np.array([[np.float(x) for x in row] for row in rows if isfloats(row)])
#         return A
    
    
d = {'a': 3.141592653589793,
 'b': True,
 'c': {'c1': np.array([[2.3, 2.3],
         [2.3, 2.3]]), 'c2': {'c21': None, 'c22': np.inf,'cjhekjhlfkqjezhl':'C:/Users/Simon/Documents/0_BEC/1_DataDict_TREATMENT/dada/handle/DataDict2.py','fekjhlkejjnfker fjior':np.diag([1,2,3,4,5])}},
 'd': 'Bonjour',
 'e': Fraction(1, 7)}
D = DataDict(d)
# print(DataDict(d).merge({'c':{'c2':{'c23':0}}}))