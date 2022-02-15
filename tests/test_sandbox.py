# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:26:03 2017

@author: Peter Modregger

Updated for iterlib v 5.3

Example for numerical simulation for the iterative reconstruction. It should
take about 15 secs to show the results. Script has to be run in the
\example\ subfolder to make the iterlib available.

Change iter_dict['callback'] = 1 to see the progress of iteration

"""

# import python stuff
import numpy as np
import numpy.matlib
import matplotlib.pyplot as pl
import time
import pm

# import the iterlib 
import sys
sys.path.append('../libs/')
import iterlib
import phantom

# set some defaults I like
np.random.seed(seed=2)
pl.rcParams['image.interpolation'] = 'none'
pl.rcParams['image.cmap'] = 'gray'
pl.rcParams.update({'font.size': 8})
pl.rcParams['savefig.bbox'] = 'tight'
pl.rcParams['image.aspect'] = 'equal'

#%% some functions
def gauss(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

def countstat(I,noiseon=1):
    "Usage: I+noise, noise = pm.countstat(I)"
    if len(np.asarray(I))>1:
        I[I<0] = 0 # I might be negative and the sqrt would give NaN results
    if noiseon:    
        Inoise = np.random.poisson(lam=I).astype('float')
    else:
        Inoise = I        
    return Inoise, I - Inoise


#%% tomo scan parameters
dr = 2500 # dynamic range for poisson statistics (higher = better statistics)
Ndet = 180 # number of detector pixels
Ntheta = Ndet//2 # number of projections
M = 1 # number of scan points on IC during tomo (right now just M=1 works)
RA = Ndet/2-0.5 # location of rotation axis in pixels
m = -4.0 # point(s) on the IC during tomo scan
mo_org = (0.5-np.random.rand(Ntheta,M))*.5 # lateral mask position offset
mr_org = (0.5-np.random.rand(Ndet,M))*1.5 # artificial ring introduction
theta_deg = np.linspace(0,360,Ntheta,endpoint=False)
theta = np.deg2rad(theta_deg)

# IC flat parameters
Mflat = 33 # number scan points on IC flat
mflat = np.linspace(-15,15,Mflat) # scan points on "measured" IC flat. uses values of order 1 instead of physical mask positions
fmu = 2*(np.random.rand(Ndet)-0.5)*0.2*1 # local offset of flat IC (i.e., mask imperfections)
fsi = 2*((np.random.rand(Ndet)-0.5)*0.1*1+2) # local width of flat IC (i.e., mask imperfections)

# some flips
noiseon = 1 # noise during tomoscan?
flatnoiseon = 1 # noise during IC flat scan
gamma_abs = 1 # use abs data?
gamma_phs = 1 # use phase data?
gamma_sim = 10 # gamma used in forward model

#%% defining the sample slice
# absorption slice
orga = phantom.phantom(Ndet,ellipses=[[1.2,0.2,0.2,0,-0.4,0],
                                  [1.0,0.2,0.2,-.4,0.4,0],
                                  [0.8,0.2,0.2,.4,0.4,0]])*2**2/Ndet/10*2*2*1
# phase slice
orgp = phantom.phantom(Ndet,ellipses=[[1.4,0.2,0.2,0,-0.4,0],
                                  [1.0,0.2,0.2,-.4,0.4,0],
                                  [0.6,0.2,0.2,.4,0.4,0]])*2**2/Ndet/10*2*2*1/1
# for true single shot uncomment the following line
orgp = gamma_sim * orga
    
  
#%% using the forward model to generate inogram
asino = iterlib.radon_numba_linear(orga,theta,RA);
asino = np.exp(-gamma_abs*asino) # absorption contrast sino
tsino = iterlib.radon_numba_linear(orgp,theta,RA);
dsino = np.zeros(asino.shape)
dsino[1:,:] = np.diff(tsino,1,0)*gamma_phs # phase contrast sino

msim = np.linspace(-15,15,1001) # points on IC for forward simulation (should be higher sampling that "measured" flat IC
fsim = np.zeros((Ndet,len(msim)))
ICsamp = np.zeros((Ndet,Ntheta,M))
for isp in range(Ndet):
    fsim[isp,:] = gauss(msim,fmu[isp],fsi[isp]) # ICs are simple Gaussians
    fsim[isp,:] = fsim[isp,:]/fsim[isp,:].max()*dr # scaling in case that noise is turned on
    for ist in range(Ntheta):
        ICsamp[isp,ist,:] = asino[isp,ist]*np.interp(m-mo_org[ist,:]-mr_org[isp,:],msim-dsino[isp,ist],fsim[isp,:])

if noiseon: # add photon shot noise if turned on
    ICsamp, dummy = countstat(ICsamp)

#%% "measured" flat IC
ICflat = np.zeros((Ndet,Mflat))
for isp in range(Ndet):
    ICflat[isp,:] = gauss(mflat,fmu[isp],fsi[isp]) # ICs are simple Gaussians
    ICflat[isp,:] = ICflat[isp,:]/np.max(ICflat[isp,:])*dr # scaling in case that noise is turned on

if flatnoiseon: # add photon shot noise if turned on
    ICflat, dummy = countstat(ICflat)

        
#%% plotting the sinograms
pl.subplot(221)
pl.imshow(asino,aspect='auto');pl.colorbar();pl.title('abs sino')
pl.subplot(222)
pl.imshow(dsino,aspect='auto');pl.colorbar();pl.title('ref sino')

pl.subplot(223)
pl.plot(mflat,ICflat.transpose());pl.title('flat ICs')
pl.subplot(224)
pl.imshow(ICsamp[:,:,0],aspect='auto');pl.colorbar();pl.title('sinogram')
pl.show(); print('')


#%% single shot
gamma_ss = gamma_sim # gamma used for single shot reconstruction
rec_ss = iterlib.shiftimg(iterlib.single_shot(m,mflat,ICflat,np.squeeze(ICsamp),theta_deg,gamma_ss),0.5,0.5)
pl.imshow(rec_ss,vmin=0);pl.colorbar();pl.title('single shot');pl.show();print('')

  
#%%  defining the dictionary for iterative reconstruction

# default iter dict
iter_dict = iterlib.iter_parameters()

# parameters for iteration
iter_dict['maxiter'] = 1000
iter_dict['m'] = 100 # number of matrix corrections for BFGS
iter_dict['maxls'] = 3 # number of line search steps for BFGS
iter_dict['factr'] = 1e9*1 # 1e9 works fine and fast
iter_dict['callback'] = 0 # set to 1 to get update after each iteration step

# signals used for forward model
iter_dict['gamma_abs'] = gamma_abs # turn on abs signal in forward model?
iter_dict['gamma_phs'] = gamma_phs # turn on phase signal in forward model?

# radon look up table
iter_dict['Radon_algo'] = 'ind' # algorithm for sinogram generation
iter_dict['RA'] = RA # this is needed if gradradon are not given
iter_dict['theta'] = theta

# determine which values to retrieve
iter_dict['do_rec_slice_abs'] = 1
iter_dict['do_rec_slice_phs'] = 0
iter_dict['do_rec_mo'] = 1 # offset
iter_dict['do_rec_mr'] = 1 # ring artefact

# start values
iter_dict['slice_abs'] = [] # leave empty to not provide a starting/fixed value
iter_dict['mo'] = []        # start will be constructed out of these if appropriate
iter_dict['mr'] = []
iter_dict['gamma'] = gamma_sim

# noise suppression parameters
iter_dict['lambda1'] = 5e-3
# iter_dict['lambda1'] = 0*5e-3
iter_dict['delta1'] = 100 # set high to make noise reduction L2 norm
iter_dict['lambda2'] = 0#1 
iter_dict['delta2'] = 0.2
iter_dict['lambdatv'] = 0e-4




#%% do the iteration
iter_dict['factr'] = 1e6*1 # 1e9 works fine and fast
iter_dict['lambda1'] = 0e-3
iter_dict['weights'] = 1 # leave at 1. doesnt make a huge difference for EI
rec_dict, funcs = iterlib.iter_single_shot(mflat,ICflat,m,ICsamp,iter_dict)
rec_abs = rec_dict['rec_slice_abs']
print('NMSE - weight = 1: ',pm.nmse(orga, rec_abs))
pl.subplot(121)
pl.imshow(rec_abs)
print('\n\n\n')

iter_dict['factr'] = 1e6*1 # 1e9 works fine and fast
iter_dict['weights'] = 1/np.sqrt(ICsamp)
iter_dict['lambda1'] = 0e-5
rec_dict, funcs = iterlib.iter_single_shot(mflat,ICflat,m,ICsamp,iter_dict)
rec_abs = rec_dict['rec_slice_abs']
print('NMSE - weight = 1/sqrt: ',pm.nmse(orga, rec_abs))
pl.subplot(122)
pl.imshow(rec_abs)

#%% show the results
rec_abs = rec_dict['rec_slice_abs']
rec_phs = rec_dict['rec_slice_phs']
rec_mo = rec_dict['rec_mo']
rec_mr = rec_dict['rec_mr']

pl.subplot(221)
pl.imshow(rec_ss);
pl.colorbar()
pl.title('single shot')

pl.subplot(222)
pl.imshow(rec_abs);pl.colorbar()
pl.title('iterative reconstruction')

pl.subplot(223)
pl.plot(mo_org,rec_mo,'.')
pl.title('mask offset $m_o$')
pl.xlabel('original')
pl.ylabel('retrieved')

pl.subplot(224)
pl.plot(mr_org,rec_mr,'.')
pl.title('ring offstt $m_r$')
pl.xlabel('original')
pl.ylabel('retrieved')

#pl.savefig('numsim_example.png')

pl.show(); print('')

