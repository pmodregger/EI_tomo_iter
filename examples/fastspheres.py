# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:26:03 2017

@author: Peter Modregger

Updating for iterlib v 5.5

Example for using iterative tomographic reconstruction. The sample is composed
of three plastic spheres. The data was cut and reordered according to the
requirements of the iterative procedure. It will take about 20 seconds to 
finish
"""


# import python stuff
import numpy as np
import matplotlib.pyplot as pl
import time

# import the iterlib 
import sys
sys.path.append('../libs/')
import iterlib

# set some defaults I like
np.random.seed(seed=2)
pl.rcParams['image.interpolation'] = 'none'
pl.rcParams['image.cmap'] = 'gray'
pl.rcParams.update({'font.size': 6})
pl.rcParams['savefig.bbox'] = 'tight'
pl.rcParams['image.aspect'] = 'equal'

tt = time.time()
print('This is an example for iterative reconstruction.')
print('It will take about 25 seconds to finish.')

#%% load data 
data = np.load('fastspheres.npz') # this data was prepared

ICsamp = data['ICsamp'] # the sample IC. Indexing: ICsamp[Nt,Ntheta,M]
ICflat = data['ICflat'] # the flat field IC. Indexing: ICflat[Nt,Mflat]
mflat = data['mflat'] # scan positions of the flat fielf IC. Use "natural units" like np.linspace(-15,15,33) instead of actual positions in um!
m0 = data['m0'] # the scan position during tomographic scan 
RA = data['RA'] # rotation axis in pixels. Subpixel precision is allowed.
theta = np.deg2rad(data['theta']) # the projection angles

(Nt,Ntheta,M) = ICsamp.shape
(Nt,Mflat) = ICflat.shape  

print('Number of pixels (Nt): ', Nt)
print('Number of projections (Ntheta): ', Ntheta)
print('Number of scan points for flat IC (Mflat): ',Mflat)
print('Number of scan points on IC during tomo scan:',M)
print('Location of the rotation axis: ',RA)

#%% Parameters and lookup tables are given to the iteration function in the iter_dict dictionary
iter_dict = iterlib.iter_parameters() # gives back a default dictionary with all necessary keywords

#% Adjust the iteration dictionary to your needs
iter_dict['maxiter'] = 1000 # max number of iteration steps
iter_dict['factr'] = 1e8 # adjust stopping condition for the iteration. 1e9 works fine for most cases. 1e0 is for high precision. 
iter_dict['callback'] = 0 # get updates after each iteration step (0: no, 1: yes)

iter_dict['do_rec_slice_abs'] = 1 # reconstruct the absorption slice?
iter_dict['do_rec_slice_phs'] = 0 # reconstruct the phase slice? For abs=1 & phs=0 reconstructs combined slice. For abs=1 & phs = 1 reconstructs separate contrasts
iter_dict['do_rec_mo'] = 1 # reconstruct the offset for lateral mask position?
iter_dict['do_rec_mr'] = 1 # reconstruct the offset for ring supression?

iter_dict['gamma'] = 5 # proportionality factor between absorption and phase contrast for combined reconstructions
     
iter_dict['lambda1'] = 4e-3 # factor for noise regularisation of the absorption or combined contrast slice
iter_dict['delta1'] = 10 # switch for noise regularisation. values lower than reconstructed values for L1 norm. values for higher than reconstructed values for L2 norm. (Use L2 norm)
iter_dict['lambda2'] = 0e-3 # factor for noise regularisation of phase contrast slice
iter_dict['delta2'] = 10 # switch for noise regularisation. values lower than reconstructed values for L1 norm. values for higher than reconstructed values for L2 norm. (Use L2 norm)

iter_dict['RA'] = RA # the rotation axis
iter_dict['theta'] = theta # projection angles

#%% Calling the iteration function
rec_dict, funcs = iterlib.iter_single_shot(mflat,ICflat,m0,ICsamp,iter_dict)
    
#%% Extracting the data from the resulting dictionary
rec_abs = rec_dict['rec_slice_abs']
rec_phs = rec_dict['rec_slice_phs']
rec_mo = rec_dict['rec_mo']
rec_mr = rec_dict['rec_mr']

#%% Using standard single shot approach for comparison
rec_ss = iterlib.single_shot(m0,mflat,ICflat,np.squeeze(ICsamp),np.rad2deg(theta),iter_dict['gamma'],dorec=1)

#%% plotting the result
pl.rcParams.update({'font.size': 10})
pl.subplot(221)
pl.imshow(rec_ss/iter_dict['gamma'])
pl.title('standard single shot')
pl.colorbar()
pl.subplot(222)
pl.imshow(rec_abs)
pl.colorbar()
pl.title('joined reconstructed slice')
pl.subplot(223)
pl.plot(theta,rec_mo)
pl.title('retrieved mask position offset')
pl.xlabel('projection angle $\\theta$ / deg.')
pl.ylabel('$m_o(\\theta)$')
pl.subplot(224)
pl.plot(rec_mr) 
pl.title('retrieved ring suppression offset')
pl.xlabel('position / pixel')
pl.ylabel('$m_r(t)$')

#pl.savefig('fastspheres.png')
pl.show()

print('time elapsed: ',time.time()-tt,'s')