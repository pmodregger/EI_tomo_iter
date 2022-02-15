# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 10:55:05 2021

@author: Peter

Testing the standard tomographic reconstruction part of iterlib

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

#%% original
Ndet = 360
RA = Ndet//2+.4
theta_deg = np.linspace(0,180,Ndet//2,endpoint=False)
theta = np.deg2rad(theta_deg)
Ntheta = len(theta)

org = shepp_logan_phantom()
org = rescale(org, scale=Ndet/400, mode='reflect')
expsino = iterlib.radon_numba_linear(org,theta,RA)

#%% iteration dictionary
iter_dict = iterlib.iter_parameters()
iter_dict['do_rec_slice_abs'] = 1 # retrieve slice?
iter_dict['RA'] = RA # this is needed if gradradon_x are not given
iter_dict['theta'] = theta # rotation angles in radians
iter_dict['maxiter'] = 50 # max number of iteration steps
iter_dict['m'] = 100 # number of matrix corrections for BFGS; try m=100
iter_dict['maxls'] = 3 # number of line search steps for BFGS; try maxls=2,3
iter_dict['factr'] = 1e9 # 1e9 works fine and fast
iter_dict['callback'] = None
# iter_dict['callback'] = 'callback' # show updates after each iteration step

#%% do the iteration and show result
rec_dict, funcs = iterlib.iter_tomo_rec(expsino,iter_dict)

rec1 = rec_dict['rec_slice_abs']

pl.subplot(131)
pl.imshow(org);
pl.subplot(132)
pl.imshow(rec1);
pl.subplot(133)
pl.imshow(np.abs(org-rec1),vmin=0,vmax=np.max(org)); 
print('NMSE: ',np.sum( (org-rec1)**2)/np.sum(org))

pl.show()

# TIMINGS
# Ndet =  90, factr = 1e9,  1 s, -> 0.426 s > 0.427 s
# Ndet = 180, factr = 1e9,  6 s, -> 2.78 s -> 3.14 s
# Ndet = 200, factr = 1e9,          4 s
# Ndet = 360, factr = 1e9, 40 s, -> 19.1 s -> 22.5 s
# Ndet = 720, factr = 1e9, 260 s, -> 128 s
# Ndet = 1440, factr = 1e9,  -> 1017 s