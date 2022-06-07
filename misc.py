# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:27 2022, @author: Simon, @version: 1.0
"""

import winsound
import numpy as np
from collections.abc import Iterable

import sfloat

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colorbar as clb
from cycler import cycler

def bip(n,duration=500):
    winsound.Beep(int(440*2**((n+2)/12)),int(duration))
    
def wmean(X,**kwargs):
    if len(X)==1: return sfloat(X[0],s=X[0].s,**kwargs)
    a, b = np.sum([x/x.s**2 for x in X]), np.sum([1/x.s**2 for x in X])
    m, s = a/b, 1/np.sqrt(b)
    birge_ratio = np.sqrt(np.sum([(x-m)**2/x.s**2 for x in X])/(len(X)-1))
    return sfloat(m,s={True:1,False:birge_ratio}[birge_ratio>1]*s,**kwargs)

def medmask(x,delta=10):
    x,med = np.array(x),np.median(x)
    return abs(x-med) < delta*np.median(abs(x-med))

def arrayify(*args,mask=None,sort=False,dtype=None):
    if mask is None: mask = np.ones(len(args[0]),dtype=bool)
    if sort: order = np.argsort(np.array(args[0])[mask])
    else:    order = np.ones(len(np.array(args[0])[mask]),dtype=bool)
    L = [np.array(A,dtype=dtype)[mask][order] if A is not None 
         else None for A in args]
    return L[0] if len(L)==1 else L
    # if (dtype is sfloat) or (sfloat in [type(x) for x in X]): X
    
def encode(x):
    if isinstance(x,sfloat): return x.__dict__
    if isinstance(x,dict): return {k:encode(v) for k,v in x.items()}
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.int32): return int(x)
    else: return x

def matrix_test(M,cmap='jet'):
    if np.max(abs(M))<1E-14: 
        print('Matrice nulle')
        return
    plt.figure()
    plt.imshow(np.array(M,dtype=float),interpolation='nearest',cmap=cmap)
    plt.colorbar()  
    
def style(style='default'):
    axcol = '0.2'
    styles = {'default':{'font.size':12,'font.sans-serif':'Helvetica',
                          'axes.prop_cycle':cycler('color',['midnightblue',
                          'crimson','goldenrod','#e6c89a','#e68c7a','#8d9db3']),
                          'lines.linewidth':1.8,'lines.markersize': 5.0,
                          'axes.labelsize': 14,'axes.linewidth':0.9,
                          'figure.subplot.wspace': 0.04,
                          'figure.subplot.hspace': 0.1,'axes.labelcolor':axcol,
                          'axes.edgecolor':axcol,'ytick.color':axcol,
                          'xtick.color':axcol,'text.color':axcol},
              'no-border':{'xtick.major.size':0,'ytick.major.size':0,
                          'xtick.minor.size':0,'ytick.minor.size':0,
                          'ytick.labelsize':0,'xtick.labelsize':0,
                          'axes.spines.bottom':False,'axes.spines.left':False,
                          'axes.spines.right':False,'axes.spines.top':False}}
    plt.rcParams.update(styles[style])
# style()

def cmap(s):
    COLORS = {'b':'midnightblue','r':'crimson','y':'goldenrod',
              'w':'#f0efeb','k':'0.1','g':'0.5'}
    return LinearSegmentedColormap.from_list('',[COLORS[c] for c in s])

def listify(*args,itemsAreIterable=False):
    _ = []
    for arg in args:
        if arg is None: _.append(None)
        elif not isinstance(arg,Iterable) or type(arg) is str: _.append([arg])
        elif itemsAreIterable and not isinstance(arg[0],Iterable): _.append([arg])
        else: _.append(arg)
    return _
        
class Multiplot():    
    def __init__(self,x,y,yerr=None,xerr=None,z=None,xlabel=None,ylabel=None,
                  zlabeldict=None,title='',cmap=cmap('bw'),marker='o',
                  linestyle='--',figsize=(10,6),tight_layout=True):
        x,y,xerr,yerr,xlabel,ylabel = listify(x,y,xerr,yerr,xlabel,ylabel,
                                              itemsAreIterable=True)
        self.fig,axs = plt.subplots(nrows=len(y),ncols=len(x),figsize=figsize)
        self.axs = np.reshape(axs,np.size(axs))
        plt.title(title)
        subplots = [(i,xi,j,yj) for j,yj in enumerate(y) for i,xi in enumerate(x)]
        for k,((i,xi,j,yj),ax) in enumerate(zip(subplots,self.axs)):
            zbins = np.sort(np.unique(z)) if z is not None else [None]
            for l,zbin in enumerate(zbins):
                M = z==zbin if zbin is not None else np.ones(len(xi),dtype=bool) 
                xx,yy = xi[M],yj[M]
                xxerr = xerr[i][M] if xerr is not None else None
                yyerr = yerr[j][M] if yerr is not None else None
                label = zlabeldict[zbin] if zlabeldict is not None else None
                color = cmap(l/len(zbins))
                ax.errorbar(*arrayify(xx,yy,yyerr,xxerr,sort=True),label=label,
                        color=color,marker=marker,linestyle=linestyle,zorder=-l)
            if xlabel is not None and j==(len(y)-1): ax.set_xlabel(xlabel[i])
            if ylabel is not None and i==0: ax.set_ylabel(ylabel[j])
            if j<(len(y)-1): ax.set_xticks([])#,[])
            if i>0: ax.set_yticks([])#,[])
            if zlabeldict is not None and (i==len(x)-1 and j==0):
                if len(zlabeldict)<6: plt.legend()
                else: 
                    colormap_legend(plt.gcf(),plt.gca(),zbins,zlabeldict,cmap)
                    tight_layout = False
        if tight_layout: plt.tight_layout()
        
    # @classmethod
    # def from_dataframe(cls,data,xkey,ykey):
    #     data,xkey,ykey = listify(data,xkey,ykey,itemsAreIterable=(True))
    #     x,y = [],[]
    
    
def colormap_legend(fig,axis,zbins,zlabeldict,cmap,loc=1):
    position = {1:[0.64,0.78,0.2,0.02]}
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=0,vmax=1))
    # sm._A = []
    axis.add_patch(Rectangle((0.63,0.8),0.325,0.16,fill=True,transform=axis.transAxes))
    # axis.add_patch(FancyBboxPatch((0.63,0.8),0.325,0.16,transform=axis.transAxes))
    ax_cb = fig.add_axes(position[loc])
    cb = clb.Colorbar(ax_cb,sm,orientation='horizontal')
    cb.set_ticks([0,1])
    cb.set_ticklabels([zlabeldict[zbins[0]],zlabeldict[zbins[-1]]])
    if 'label' in zlabeldict.keys(): 
        ax_cb.set_title(zlabeldict['label'])