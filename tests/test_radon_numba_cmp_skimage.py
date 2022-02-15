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
import sys
sys.path.append('../libs/')
import iterlib
importlib.reload(iterlib)
pl.rcParams.update({'font.size': 5})

def nmse(data1,data2):
    return np.mean((data1-data2)**2)/np.mean(data1**2)

#%% original
Ndet = 90
RA = Ndet//2+0.0
theta_deg = np.linspace(0,360,Ndet//1,endpoint=False)
theta = np.deg2rad(theta_deg)
Ntheta = len(theta)

org = shepp_logan_phantom()
org = rescale(org, scale=Ndet/400, mode='reflect')


expsino_ski = radon(org,theta_deg,RA)
expsino_rnl = iterlib.radon_numba_linear(org,theta,RA)


pl.subplot(221)
pl.imshow(expsino_ski)
pl.colorbar()
pl.subplot(222)
pl.imshow(expsino_rnl)
pl.colorbar()

pl.subplot(223)
pl.imshow(expsino_rnl-expsino_ski)
pl.colorbar()

print(nmse(expsino_rnl,expsino_ski))
