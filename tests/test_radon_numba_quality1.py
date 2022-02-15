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


#%% original
Ndet = 180
RA = Ndet//2+0
theta_deg = np.linspace(0,360,Ndet//1,endpoint=False)
theta = np.deg2rad(theta_deg)
Ntheta = len(theta)

org = shepp_logan_phantom()
org = rescale(org, scale=Ndet/400, mode='reflect')

org = np.zeros((Ndet,Ndet))
org[Ndet//4,Ndet//3] = 1

expsino_ski = radon(org,theta_deg,RA)
expsino_rnl = iterlib.radon_numba_linear(org,theta,RA)


pl.subplot(221)
pl.imshow(expsino_ski)
pl.title(Ndet)
pl.colorbar()
pl.axis('equal')
pl.subplot(222)
pl.plot(np.sum(expsino_ski,0))
# pl.ylim((0,2))


pl.subplot(223)
pl.imshow(expsino_rnl)
pl.title(Ndet)
pl.colorbar()
pl.axis('equal')
pl.subplot(224)
pl.plot(np.sum(expsino_rnl,0))
# pl.ylim((0,2))

# pl.colorbar()
# pl.subplot(223)
# # pl.imshow(expsino_org-expsino_lib)
# pl.colorbar()
# pl.axis('equal')
# pl.subplot(224)
# # pl.semilogy(Ndetall,maxerror)
# pl.show()


