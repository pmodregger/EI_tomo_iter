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
# import iterlib_5_2
import iterlib as iterlib
importlib.reload(iterlib)
pl.rcParams.update({'font.size': 5})


#%% original
Ndet = 360
RA = Ndet//2+0.
theta_deg = np.linspace(0,360,Ndet//2,endpoint=False)
theta = np.deg2rad(theta_deg)
Ntheta = len(theta)

org = shepp_logan_phantom()
org = rescale(org, scale=Ndet/400, mode='reflect')

t1_p = time.perf_counter()
expsino= iterlib.radon_numba_linear(org,theta,RA)
t2_p = time.perf_counter()
print(Ndet, "-->", t2_p-t1_p)

pl.imshow(expsino)