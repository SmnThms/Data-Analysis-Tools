# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:27 2022, @author: Simon, @version: 1.0
"""

from data import Data
import numpy as np

class DS(np.ndarray):
    def __new__(cls,input_array):
        return np.asarray(input_array).view(cls)
        # obj.__keys__ = np.unique(np.concatenate([list(d.keys()) for d in obj 
        #                                          if isinstance(d,Data)]))
        # if np.any([isinstance(d,Data) for d in obj]):
        #     all_keys = [list(d.keys()) for d in obj if isinstance(d,Data)]
        #     for k in np.unique(np.concatenate(all_keys)):
        #         values = [d.get(k,np.nan) if isinstance(d,Data) 
        #                   else np.nan for d in obj]
        #         setattr(obj,k,DS(values,parent=(obj,k)))
        # elif parent is not None:
        #     setattr(obj,'__origin__',(*parent,np.result_type(np.array(input_array))))
        # return obj
        
    def __iter__(self):
        print('iter')
        yield from super().__iter__() 
    
    def __is_final_stage__(self):
        return not np.any([isinstance(x,Data) for x in self])
    
    # def __getitem__(self,index,internal=True):
    #     if not self.__is_final_stage__() or internal:
    #         return super().__getitem__(index)
        # if isinstance(index,int):
        #     indices = [index]
        # elif isinstance(index,slice):
        #     print('slicing',index)
        #     start,stop,step = index.indices(len(self))
        #     if start is None: start = 0
        #     if stop is None: stop = len(self)+1
        #     indices = np.arange(start,stop,step)
        # elif isinstance(index,np.ndarray):
        #     print('boolean indexing')
        #     if index.dtype=='bool':
        #         indices = [i for i in range(len(self)) if index[i]]
        # else: return
        # obj = DS([self[i] for i in indices])
        # if self.__is_final_stage__():
        #     obj.__parent__ = DS([self.__parent__[i] for i in indices])
        #     obj.__parent_key__ = self.__parent_key__
        # return obj
    
    def __getattr__(self,k):
        if k[:2]=='__': return self.__dict__.get(k)
        result = DS([d.get(k,np.nan) if isinstance(d,Data) else np.nan for d in self])
        if result.__is_final_stage__():
            result.__parent__ = self
            result.__parent_key__ = k
            # X.__origin__ = (self,k,np.result_type(np.array(L)))
        return result
    
    def __setattr__(self,k,v):
        if k[:2]=='__': self.__dict__[k] = v ; return
        for i,d in enumerate(self):
            if hasattr(d,k): setattr(d,k,v[i])
    
    def __array_finalize__(self,obj):
        if obj is None: return
        if isinstance(obj,DS):
            if obj.__is_final_stage__():
                self.__parent__ = obj.__parent__
                self.__parent_key__ = obj.__parent_key__
            
        # if hasattr(obj,'__extracted_from__'):
        #     if obj.__extracted_from__ is not None:
        #         parent,key,dtype = obj.__origin__
        #         setattr(parent,key,dtype.type(obj))
        #         self.__origin__ = obj.__origin__
                
                # setattr(*obj.__extracted_from__,obj)
                # self.__extracted_from__ = obj.__extracted_from__
                # self.__original_types__ = obj.__original_types__
                # self.__extracted_from__ = obj.__extracted_from__
                # setattr(*self.__extracted_from__,self)
        # else: print('sans __extr. :',obj)
        
    # def __update_the_original_data__(self,obj):
        
    
    # def __array_wrap__(self,out_arr,context=None):
    #     print('In wrap')
    #     # return super().__array_wrap__(self, out_arr, context)
    #     if hasattr(out_arr,'__extracted_from__'):
    #         if out_arr.__extracted_from__ is not None:
    #             setattr(*out_arr.__extracted_from__,out_arr)
    #             self.__extracted_from__ = out_arr.__extracted_from__
    #     return np.ndarray.__array_wrap__(self,out_arr,context)    
    #     # return super().__array_wrap__(self,out_arr,context)    
        
    # def __getslice__(self,i,j):
    #     return self
    
    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # print('In ufunc')
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, DS):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, DS):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], DS):
                print('Ohlala')
                # obj = inputs[0]
                # if hasattr(obj,'__extracted_from__'):
                #     if obj.__extracted_from__ is not None:
                #         setattr(*obj.__extracted_from__,obj)
                #         self.__extracted_from__ = obj.__extracted_from__
                # inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(DS)
                          if output is None else output)
                        for result, output in zip(results, outputs))
        # print(all([type(x) is bool for x in results[0]]))
        # print(results[0])
        if results and isinstance(results[0], DS):
            if results[0].__is_final_stage__():
                setattr(results[0].__parent__,results[0].__parent_key__,
                        results[0])
                        # np.result_type(results[0]).type(results[0]))
                        
            # if hasattr(results[0],'__origin__'):
            #     if results[0].__origin__ is not None:
                    # print(results[0].__origin__[-1])
                    # print(results[0],'len =',results[0].__len__)
                    
                    # ->
                    # parent,key,dtype = results[0].__origin__
                    # setattr(parent,key,dtype.type(results[0]))
                    # self.__origin__ = results[0].__origin__
                    
            # print('Gasp')
            # obj = results[0]
            # if hasattr(obj,'__extracted_from__'):
            #     if obj.__extracted_from__ is not None:
            #         # print(obj.__extracted_from__)
            #         setattr(*obj.__extracted_from__,obj)
            #         setattr(*obj.__extracted_from__,np.float64(obj))
            #         self.__extracted_from__ = obj.__extracted_from__
            #         self.__original_types__ = obj.__original_types__
            # results[0].info = info

        return results[0] if len(results) == 1 else results
    
    
                
    # def __str__(self):
    #     return ''.join([f'[{i}] {d}\n' for i,d in enumerate(self)])
            
    # def get(self,k):
    #     _ = self
    #     for sub_k in k.split('.'):
    #         if not hasattr(_,sub_k):
    #             return None
    #         _ = getattr(_,sub_k)
    #     return _
    
    # def keys(self):
    #     return self.__dict__.keys()
    
    def table(self,keys=[],align='c',labels={}):
        if keys==[]: keys = self.__dict__.keys()
        columns = []
        for k in keys:
            if k in labels.keys(): columns.append(labels[k])
            else: columns.append(k)
        s = '\\begin{table}\\begin{center}\\begin{tabular}' 
        s += '{|' + '|'.join([align for i in keys]) + '|} \\hline '
        s += ' & '.join(columns) + ' \\\\ \\hline '
        for d in self:
            s += ' & '.join([str(d.get(k)) for k in keys]) + ' \\\\ \\hline '
        s += '\\end{tabular}\\end{center}\\end{table}'
        print(s)
        
    def sort_by(self,key):
        return self[np.argsort(self.get(key))]
    
    # def split_by(self,key): 
    #     L = []
    #     for val in np.unique([d[key] for d in self]):
    #         L.append(self[d[key]==val for d in self])
    #     return L
    
    # def cluster_by(self,key,N=2):
    #     pass
    