# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:27 2022, @author: Simon, @version: 1.0
"""

from misc import arrayify,encode
from sfloat import sfloat

import os, json
import numpy as np
from inspect import signature
from matplotlib import pyplot as plt

from scipy.optimize import curve_fit, leastsq
from scipy.interpolate import interp1d
from scipy import odr
import scipy.stats as stats
    
ref_func = {
 'cst':lambda x,K:K*np.ones(len(x)),
 'linear':lambda x,a,b:a*x+b,
 'sin':lambda x,A,f,x0:A*np.sin(2*np.pi*f*(x-x0)),
 'exp':lambda x,A,tau:A*np.exp(-abs(x)/tau),
 'lorentz':lambda x,A,G,x0:A*(2/(G*np.pi))/(1+((x-x0)/(G/2))**2),
 'gauss':lambda x,A,G,x0:A/(G*np.sqrt(2*np.pi))*np.exp(-(x-x0)**2/(2*G**2))
 }      
     
ref_keys = {'cst':['K'],'linear':['a','b'],'sin':['A','f','x0'],
            'exp':['A','tau'],'lorentz':['A','G','x0'],'gauss':['A','G','x0']}  

ref_bounds = {'A':(0,np.inf),'G':(0,np.inf),'f':(0,np.inf)}  
    
AnalyticalCI = {
 'linear': lambda x,xi:np.sqrt(1/len(xi) \
                              + (x-np.mean(xi))**2/np.sum((xi-np.mean(xi))**2))    
    }
    
def func_sum(f1,f2):
    n = len(signature(f2).parameters) - 1
    return lambda x,*p: f1(x,*p[:-n]) + f2(x,*p[-n:])

class Fit():
    """ 
    Three ways to pass the 'func' argument:
        - a function lambda x,*p: f(x,p)
        - the degree (integer) of a polynomial
        - the name (string) of a reference function, 
          or of several reference functions separated by '+'
    """
    def __init__(self,func,x,y,xerr=None,yerr=None,p0=None,keys=None,units=None,
                 bounds=(-np.inf,np.inf),method='curve_fit',name=None,func_name='',**kwargs):
        x,y,xerr,yerr = arrayify(x,y,xerr,yerr,dtype=np.float64)
        for k in locals().copy().keys(): setattr(self,k,locals()[k])
        self.valid = True
        if type(func) is str: 
            self.func,self.keys = lambda x:0,[]
            for term in func.split('+'):
                self.func = func_sum(self.func,ref_func[term])
                self.keys += ref_keys[term]
            if p0 is None:
                if func=='linear': p0 = Fit(1,x,y).p[::-1]
                else: p0 = [1]*len(self.keys)
            self.bounds = [ref_bounds[k] if k in ref_bounds.keys() 
                            else (-np.inf,np.inf) for k in self.keys]
            self.func_name = func
        try:
            if type(func) is int: # Degree of a polynomial
                self.p,self.cov = np.polyfit(x,y,func,cov=True)
                self.func = lambda x,*p:np.polyval(p,x)
                self.keys = [f'c{n}' for n in range(func+1)[::-1]]    
                self.method = 'polynomial'
            elif method=='curve_fit':
                self.p,self.cov = curve_fit(self.func,x,y,p0=p0,sigma=yerr,bounds=bounds)
            elif method=='leastsq': # Least-square
                self.p,self.cov = leastsq(lambda p,x,y:y-self.func(x,*p),p0,args=(x,y))
            elif method=='ODR': # Orthogonal Distance Regression
                out = odr.ODR(data=odr.RealData(x,y,sx=xerr,sy=yerr),
                         model=odr.Model(lambda p,x:self.func(x,*p)),beta0=p0).run()
                self.p,self.cov = out.beta,out.cov_beta
            elif method=='MLE': # Maximum Likelyhood Estimate
                pass
            self.residuals = y - self.func(x,*self.p)
            self.dof = len(x)-len(self.p)
            self.chi2r = np.sum(self.residuals**2)/self.dof
            self.fit_std_err = stats.t.ppf(0.6827,self.dof) * np.sqrt(self.chi2r)
            self.p_std_err = np.sqrt(np.diag(self.cov))
            self.p0 = p0
            if self.keys is not None:
                for i,k in enumerate(self.keys):
                    if k!='':
                        x,s,u = self.p[i],self.p_std_err[i],None
                        if units is not None: u = units[i]
                        setattr(self,k,sfloat(x,s=s,unit=u))
            if not np.prod(np.isfinite(np.diag(self.cov))): 
                self.valid = False
        except (RuntimeError):#, TypeError, ValueError): 
            print('Fit error '+self.__repr__())
            self.valid = False
    
    def curve(self,xx=None):
        if not self.valid: return
        if xx is None: xx = np.linspace(min(self.x),max(self.x),100)
        return xx,self.func(xx,*self.p)
    
    def plot(self,xx=None,ax=None,color='#ad0019',alpha=0.5,label=None,zorder=None):
        if not self.valid: print('Fit error') ; return
        x,y = self.curve(xx)
        if ax is None: ax = plt.gca()
        ax.plot(x,y,color=color,alpha=alpha,label=label,zorder=zorder)
        if self.func_name in AnalyticalCI.keys():
            ci = self.fit_std_err * AnalyticalCI[self.func_name](x,self.x)
            plt.fill_between(x,y-ci,y+ci,color=color,
                             alpha=0.2,linewidth=0,edgecolor='',zorder=-100)        
    
    def opt_func(self,x): 
        return interp1d(*self.curve(),fill_value='extrapolate')(x)
    
    def __repr__(self): 
        return self.name if self.name is not None else ''
            
    def __str__(self):
        if not self.valid: return
        if self.__repr__()!='': print('\n'+self.__repr__())
        for k in self.keys(): print(k,';\t',self.k)
                    
    def save(self,file,folder='.'):
        d = {k:encode(v) for k,v in self.__dict__.items() if k[-4:]!='func'}
        if not os.path.isdir(folder): os.makedirs(folder)
        with open(folder+self.__repr__()+'.json','w') as f: 
            json.dump(d,f)