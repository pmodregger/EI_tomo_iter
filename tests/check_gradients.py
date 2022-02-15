# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 23:30:59 2021

@author: Peter

Update: moved checkgrad out of iterlib into this script. updated to iterlib 
version 5

"""

# import python stuff
import numpy as np
import numpy.matlib
import matplotlib.pyplot as pl
import time
from scipy import stats

# import the iterlib 
import sys
sys.path.append('../libs/')
import iterlib as iterlib
import phantom

# set some defaults I like
np.random.seed(seed=2)
pl.rcParams['image.interpolation'] = 'none'
pl.rcParams['image.cmap'] = 'gray'
pl.rcParams.update({'font.size': 10})
pl.rcParams['savefig.bbox'] = 'tight'
pl.rcParams['image.aspect'] = 'equal'


show = 1

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
Ndet = 45 # number of detector pixels
Ntheta = Ndet//2 # number of projections
M = 1 # number of scan points on IC during tomo (right now just M=1 works)
RA = Ndet/2-0.5-2 # location of rotation axis in pixels
m = -4.0 # point(s) on the IC during tomo scan
mo_org = (0.5-np.random.rand(Ntheta,M))*0.5*1 # lateral mask position offset
mr_org = (0.5-np.random.rand(Ndet,M))*1.5*1 # artificial ring introduction

# IC flat parameters
Mflat = 33 # number scan points on IC flat
mflat = np.linspace(-15,15,Mflat) # scan points on "measured" IC flat. uses values of order 1 instead of physical mask positions
fmu = 2*(np.random.rand(Ndet)-0.5)*0.2*1 # local offset of flat IC (i.e., mask imperfections)
fsi = 2*((np.random.rand(Ndet)-0.5)*0.1*1+2) # local width of flat IC (i.e., mask imperfections)

# some flips
noiseon = 0 # noise during tomoscan?
flatnoiseon = 0 # noise during IC flat scan -> this degrades the gradients
gamma_abs = 1 # use abs data?
gamma_phs = 1 # use phase data?
gamma_sim = 10 # gamma used in forward model

#%% generating the radon lookup tables
theta_deg = np.linspace(0,360,Ntheta,endpoint=False) # projection angles
theta = np.deg2rad(theta_deg)

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
if show: 
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
if show:
    
    pl.imshow(rec_ss,vmin=0);pl.colorbar();pl.title('single shot');pl.show();print('')


#%%  defining the dictionary for iterative reconstruction

# default iter dict
iter_dict = iterlib.iter_parameters()

# parameters for iteration
iter_dict['maxiter'] = 30
iter_dict['m'] = 100 # number of matrix corrections for BFGS
iter_dict['maxls'] = 10 # number of line search steps for BFGS
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
iter_dict['lambda1'] = 1e-3
# iter_dict['lambda1'] = 0*5e-3
iter_dict['delta1'] = 100 # set high to make noise reduction L2 norm
iter_dict['lambda2'] = 0#1 
iter_dict['delta2'] = 0.2
iter_dict['weights'] = 1 # leave at 1. doesnt make a huge difference for EI
# iter_dict['weights'] = np.sqrt(ICsamp) # leave at 1. doesnt make a huge difference for EI



#%% do the iteration
rec_dict, funcs = iterlib.iter_single_shot(mflat,ICflat,m,ICsamp,iter_dict)

#%% show the results
rec_abs = rec_dict['rec_slice_abs']
rec_phs = rec_dict['rec_slice_phs']
rec_mo = rec_dict['rec_mo']
rec_mr = rec_dict['rec_mr']

if show:
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


#%% checking gradients
#  funcs = [L,gradL,S,gradS_list_line2,gradLnum,Huber,gradHuber,gradSnum,gradHubernum]'

L = funcs[0]
gradLana = funcs[1]
S = funcs[2]
gradSana = funcs[3]
gradLnum = funcs[4]
Huber = funcs[5]
gradHuber = funcs[6]
gradSnum = funcs[7]
gradHubernum = funcs[8]

Nrecvalues = rec_dict['Nrecvalues']
rec_slice_abs_inds = rec_dict['rec_slice_abs_inds']

slice_condition = 0 # start of iteration
slice_condition = 1 # end of iteration
# slice_condition = 2 # sim of intermediate state

#% gradL 
if slice_condition == 0:
    test_slice = np.zeros(Nrecvalues) # starting condition
elif slice_condition == 1:
    test_slice = rec_dict['BFGS_results'][0] # end condition
elif slice_condition == 2:
     test_slice = rec_dict['BFGS_results'][0]
     test_slice[rec_slice_abs_inds] = rec_ss.flatten() # single shot result as intermediate condition

S(test_slice) # this needs to be here cause fradon and Dradon have to be updated
test_gradLana = gradLana(test_slice)
test_gradLnum = gradLnum(test_slice)

pl.subplot(211)
pl.plot(test_gradLana)
pl.plot(test_gradLnum,'g.',markersize=2)
pl.legend(('ana','num'))
pl.title('Comparing gradL ana & gradL num')
pl.subplot(212)
x = np.array((np.min(test_gradLana),np.max(test_gradLana)))
pl.plot(x,x, 'k--',linewidth=1)
pl.plot(test_gradLana,test_gradLnum,'b.',markersize=2)
pl.xlabel('gradLana')
pl.ylabel('gradLnum')
slope, intercept, r_value, p_value, std_err = stats.linregress(test_gradLana,test_gradLnum)
string =  'Regression line\n'
string += '  slope: {:.4f}\n'.format(slope)
string += '  intercept: {:.4f}\n'.format(intercept)
string += '  r value: {:.4f}\n'.format(r_value)
string += '  mean abs diff: {:.4e}'.format(np.mean(np.abs(test_gradLana-test_gradLnum)))
pl.text(np.min(test_gradLana),np.max(test_gradLnum),string,fontsize=6,verticalalignment='top')
pl.show()


#% gradS
if iter_dict['do_rec_slice_abs']:
    
    rec_slice_abs_inds = rec_dict['rec_slice_abs_inds']
    test_gradabssliceana = test_gradLana[rec_slice_abs_inds]
    test_gradabsslicenum = test_gradLnum[rec_slice_abs_inds]
    
    pl.subplot(211)
    pl.plot(test_gradabssliceana)
    pl.plot(test_gradabsslicenum,'g.',markersize=2)
    pl.legend(('ana','num'))
    pl.title('Comparing grad abs slice ana & grad abs slice num')
    pl.subplot(212)
    x = np.array((np.min(test_gradabssliceana),np.max(test_gradabssliceana)))
    pl.plot(x,x, 'k--',linewidth=1)
    pl.plot(test_gradabssliceana,test_gradabsslicenum,'b.',markersize=2)
    pl.xlabel('gradabssliceana')
    pl.ylabel('gradabsslicenum')
    slope, intercept, r_value, p_value, std_err = stats.linregress(test_gradabssliceana,test_gradabsslicenum)
    string =  'Regression line\n'
    string += '  slope: {:.4f}\n'.format(slope)
    string += '  intercept: {:.4f}\n'.format(intercept)
    string += '  r value: {:.4f}\n'.format(r_value)
    string += '  mean abs diff: {:.4e}'.format(np.mean(np.abs(test_gradabssliceana-test_gradabsslicenum)))
    pl.text(np.min(test_gradabssliceana),np.max(test_gradabsslicenum),string,fontsize=6,verticalalignment='top')
    pl.show()

# mo (mask offset position)
if iter_dict['do_rec_mo']:
    
    rec_mo_inds = rec_dict['rec_mo_inds']
    test_gradmoana = test_gradLana[rec_mo_inds]
    test_gradmonum = test_gradLnum[rec_mo_inds]
    
    pl.subplot(211)
    pl.plot(test_gradmoana)
    pl.plot(test_gradmonum,'g.',markersize=2)
    pl.legend(('ana','num'))
    pl.title('Comparing grad mo ana & grad mo num')
    pl.subplot(212)
    x = np.array((np.min(test_gradmoana),np.max(test_gradmoana)))
    pl.plot(x,x, 'k--',linewidth=1)
    pl.plot(test_gradmoana,test_gradmonum,'b.',markersize=2)
    pl.xlabel('gradmoana')
    pl.ylabel('gradmonum')
    slope, intercept, r_value, p_value, std_err = stats.linregress(test_gradmoana,test_gradmonum)
    string =  'Regression line\n'
    string += '  slope: {:.4f}\n'.format(slope)
    string += '  intercept: {:.4f}\n'.format(intercept)
    string += '  r value: {:.4f}\n'.format(r_value)
    string += '  mean abs diff: {:.4e}'.format(np.mean(np.abs(test_gradmoana-test_gradmonum)))
    pl.text(np.min(test_gradmoana),np.max(test_gradmonum),string,fontsize=6,verticalalignment='top')
    pl.show()

# mr (mask offset position)
if iter_dict['do_rec_mr']:
    
    rec_mr_inds = rec_dict['rec_mr_inds']
    test_gradmrana = test_gradLana[rec_mr_inds]
    test_gradmrnum = test_gradLnum[rec_mr_inds]
    
    pl.subplot(211)
    pl.plot(test_gradmrana)
    pl.plot(test_gradmrnum,'g.',markersize=2)
    pl.legend(('ana','num'))
    pl.title('Comparing grad mr ana & grad mr num')
    pl.subplot(212)
    x = np.array((np.min(test_gradmrana),np.max(test_gradmrana)))
    pl.plot(x,x, 'k--',linewidth=1)
    pl.plot(test_gradmrana,test_gradmrnum,'b.',markersize=2)
    pl.xlabel('gradmrana')
    pl.ylabel('gradmrnum')
    slope, intercept, r_value, p_value, std_err = stats.linregress(test_gradmrana,test_gradmrnum)
    string =  'Regression line\n'
    string += '  slope: {:.4f}\n'.format(slope)
    string += '  intercept: {:.4f}\n'.format(intercept)
    string += '  r value: {:.4f}\n'.format(r_value)
    string += '  mean abs diff: {:.4e}'.format(np.mean(np.abs(test_gradmrana-test_gradmrnum)))
    pl.text(np.min(test_gradmrana),np.max(test_gradmrnum),string,fontsize=6,verticalalignment='top')
    pl.show()

if iter_dict['lambda1'] > 0 and np.sum(np.abs(test_slice[rec_slice_abs_inds]))>0: # grad Huber of p=0 is inf
    
    test_gradHuber1ana = gradHuber(test_slice[rec_slice_abs_inds],iter_dict['delta1'])
    test_gradHuber1num = gradHubernum(test_slice[rec_slice_abs_inds],iter_dict['delta1'])
                
    pl.subplot(211)
    pl.plot(test_gradHuber1ana)
    pl.plot(test_gradHuber1num,'g.',markersize=2)
    pl.legend(('ana','num'))
    pl.title('Comparing grad Huber1 ana & grad Huber1 num')
    pl.subplot(212)
    x = np.array((np.min(test_gradHuber1ana),np.max(test_gradHuber1ana)))
    pl.plot(x,x, 'k--',linewidth=1)
    pl.plot(test_gradHuber1ana,test_gradHuber1num,'b.',markersize=2)
    pl.xlabel('gradHuber1ana')
    pl.ylabel('gradHuber1num')
    slope, intercept, r_value, p_value, std_err = stats.linregress(test_gradHuber1ana,test_gradHuber1num)
    string =  'Regression line\n'
    string += '  slope: {:.4f}\n'.format(slope)
    string += '  intercept: {:.4f}\n'.format(intercept)
    string += '  r value: {:.4f}\n'.format(r_value)
    string += '  mean abs diff: {:.4e}'.format(np.mean(np.abs(test_gradHuber1ana-test_gradHuber1num)))
    pl.text(np.min(test_gradHuber1ana),np.max(test_gradHuber1num),string,fontsize=6,verticalalignment='top')
    pl.show()

