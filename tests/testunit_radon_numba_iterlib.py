# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 10:55:05 2021

@author: Peter

Testing the radon numba linear function in iterlib with 5.0 version as standard

Problems with: Ndet=50, Ntheta=16 (0,360), RA=24.69

"""

import numpy as np
import pylab as pl
from numba import jit
from scipy import optimize
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
import time
import importlib
import iterlib
importlib.reload(iterlib)
pl.rcParams.update({'font.size': 5})

# this is the 5.0 version, which will serve as a gold standard
@jit(nopython=True,fastmath=True)
def radon_numba_linear(data,theta,RA): # radon transform
    """
    USAGE:
        out = radon_numba_linear(data,theta,RA)

        theta = in radians
        RA         
        data = [Ndet,Ndet] - slice
        out = [Ndet,Ntheta] - sino
    """
    # print('old func',func)
    def calc_t(isi,Ndet,RA):
        "Calculate point on sinus curve"
        isy, isx = isi//Ndet, isi%Ndet
        xs = isx-Ndet/2+0.5
        ys = isy-RA
        r = np.sqrt(xs**2+ys**2)
        phi = np.arctan2(ys,xs)
        return r*np.cos(theta+phi)+RA, isy, isx

    Ndet = data.shape[0]
    Ntheta = len(theta)
    n = 1.5 # this was found empirically, matching astra radon the best
    
    out = np.zeros((Ndet,Ntheta))

    for isi in range(Ndet*Ndet):
        
        t, isy, isx = calc_t(isi,Ndet,RA)
        a = np.all( (t>0) * (t<(Ndet-2)) )
        
        if a:
            for ist in range(Ntheta):
                tt = np.int32(t[ist])
           
                d1n = (t[ist]-tt)**n
                d2n = (tt+1-t[ist])**n
                     
                b = 1/(d1n+d2n)
                out[tt,ist] += data[isy,isx]*b*d2n
                out[tt+1,ist] += data[isy,isx]*b*d1n
            
    return out


#%% original

Ndetall = np.arange(30,1440,10)

# Ndetall = np.array(90,ndmin=1)
maxerror = np.zeros(len(Ndetall))
for isn in range(len(Ndetall)):

    Ndet = Ndetall[isn]
    RA = Ndet//2+round((np.random.rand()-0.5)*3,2)
    theta_deg = np.linspace(0,360,Ndet//np.random.randint(1,5),endpoint=False)
    theta = np.deg2rad(theta_deg)
    Ntheta = len(theta)
    
    org = shepp_logan_phantom()
    org = rescale(org, scale=Ndet/400, mode='reflect')
    expsino_org = radon_numba_linear(org,theta,RA)
    expsino_lib = iterlib.radon_numba_linear(org,theta,RA)
    
    maxerror[isn] = np.max(np.abs(expsino_lib-expsino_org))
    
    pl.subplot(221)
    pl.imshow(expsino_org)
    pl.title(Ndet)
    pl.colorbar()
    pl.axis('equal')
    pl.subplot(222)
    pl.imshow(expsino_lib)
    pl.title(round(RA,2))
    pl.colorbar()
    pl.axis('equal')
    pl.subplot(223)
    pl.imshow(expsino_org-expsino_lib)
    pl.colorbar()
    pl.axis('equal')
    pl.subplot(224)
    pl.semilogy(Ndetall,maxerror)
    pl.show()
    
    
