# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:27 2022, @author: Simon, @version: 1.0
"""

import numpy as np
from matplotlib import pyplot as plt

class sfloat(float):
    decimal_exponent = 'x10^' # 'e'
    decimal_separator = '.'
    default_sdigits = 1
        
    def __init__(self,x,s=0,unit=None,sdigits=default_sdigits,**kwargs):
        self.s = float(abs(s))
        for k,v in {'unit':unit,'sdigits':sdigits,**kwargs}.items():
            setattr(self,k,v)
    
    def __str__(self):        
        unc,power,unit = '','',''
        if self.unit is not None: unit = ' ' + self.unit
        if self==0: e_x = 0
        if self!=0: 
            e_x = int(np.floor(np.log10(abs(self))))
            if e_x!=0: power = sfloat.decimal_exponent + str(e_x)
        if self.s==0: return f'{self*10**-e_x}' + power + unit
        e_s = int(np.floor(np.log10(abs(self.s))))
        if self==0:
            if e_s!=0: power = sfloat.decimal_exponent + str(e_s)
            val = ''.join(['0' for i in range(self.sdigits)])
            unc = str(round(abs(self.s)*10**(-e_s+self.sdigits-1)))
        if self!=0:
            val = str(round(self*10**(-min(e_s,e_x)+self.sdigits-1)))
            unc = str(round(abs(self.s)*10**(-min(e_s,e_x)+self.sdigits-1)))
        if not (self.sdigits==1 and (e_s>=e_x or (e_s<e_x and self==0))):
            i = {True:1,False:2}[self>=0]
            val = sfloat.decimal_separator.join([val[:i],val[i:]])
        if self.sdigits>1:
            if e_s==e_x:
                unc = sfloat.decimal_separator.join([unc[0],unc[1:]])
            if e_s>e_x or (e_s<e_x and self==0):
                i = self.sdigits - 1
                unc = sfloat.decimal_separator.join([unc[:-i],unc[-i:]])    
        return val + '(' + unc + ')' + power + unit
        
    def propagate(self,func,x):
        x, u2 = np.array(x), 0
        e_i, grad = np.diag(np.ones(x.size)), np.zeros(x.size)
        for i in range(x.size):
            if isinstance(x[i],sfloat):
                if x[i].s!=0:
                    dx = abs(x[i].s)*e_i[:,i]
                    grad = (func(x+dx) - func(x-dx)) / (2*dx)
                    u2 += grad**2 * x[i].s**2
        return sfloat(func(x),s=np.sqrt(u2))
    
    def plot(self,color='0.5',label=None):
        x,y = np.array(plt.xlim()),self*np.ones(2)
        plt.plot(x,y,color=color,label=label)
        plt.fill_between(x,y-self.s,y+self.s,color=color,alpha=0.1,linewidth=0)
    
    @classmethod
    def fromDict(cls,D):
        try:
            if not np.all([hasattr(D,k) for k in ['x','s','sdigits']]):
                raise Exception
        except Exception: 
            print('Incompatible input for type sfloat')
        x = D['x']
        D.pop('x')
        return cls(x,**D)
