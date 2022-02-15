# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:26:03 2017

@author: Peter Modregger

Iterlib 4.0 still uses radon_inds
"""

from __future__ import print_function, division
import numpy as np
import pylab as pl
from scipy import optimize
import time
from scipy import stats
from scipy import interpolate
from skimage.transform import radon, iradon


def iter_parameters():
    """
    USAGE:
        iter_dict = iterlib.iter_parameters
        
    DESCRIPTION:
        returns a default dictionary of parameters for the iteration
    
    """

    iter_dict = {}
    
    # parameter for the l-bfgs-b minimisation
    iter_dict['maxiter'] = 1000 # max number of iteration steps
    iter_dict['m'] = 1000 # number of matrix corrections for BFGS; try m=100,1000
    iter_dict['maxls'] = 3 # number of line search steps for BFGS; try maxls=2,3
    iter_dict['factr'] = 1e9 # 1e9 works fine and fast
    iter_dict['callback'] = 0 # show updates after each iteration step
    
    # switches for different retrievable values
    iter_dict['do_rec_slice_abs'] = 1 # retrieve absorption slice?
    iter_dict['do_rec_slice_phs'] = 0 # 0 -> single shot, 1 -> two shots
    iter_dict['do_rec_mo'] = 0 # retrieve mask offset position?
    iter_dict['do_rec_mr'] = 0 # retrieve ring artefact suppression position?
    iter_dict['do_rec_icfac'] = 0 # exposure time factor between IC flat & IC samp
    iter_dict['do_rec_gamma'] = 0 # retrieve gamma (doesnt work very well)
    iter_dict['do_rec_flatm'] = 0 # (dont use this)
    
    # start or fixed values for retrievable values (works fine without giving starting values)
    iter_dict['slice_abs'] = [] # leave empty to not provide a starting/fixed value
    iter_dict['slice_phs'] = [] # start will be constructed out of these if appropriate
    iter_dict['mo'] = []        
    iter_dict['gamma'] = 1
    iter_dict['icfac'] = 1
    iter_dict['mr'] = []
    iter_dict['flatm'] = []
    
    # lower & upper bounds for retrievable values
    iter_dict['slice_abs_lbound'] = 0
    iter_dict['slice_abs_ubound'] = np.inf
    iter_dict['slice_phs_lbound'] = 0
    iter_dict['slice_phs_ubound'] = np.inf
    iter_dict['mo_lbound'] = -1
    iter_dict['mo_ubound'] = 1
    iter_dict['icfac_lbound'] = 1e-8
    iter_dict['icfac_ubound'] = 2
    iter_dict['gamma_lbound'] = 1e-8
    iter_dict['gamma_ubound'] = np.inf
    iter_dict['fmo_lbound'] = -1
    iter_dict['fmo_ubound'] = 1
    iter_dict['flatm_lbound'] = -1
    iter_dict['flatm_ubound'] = 1
    
    # parameters for foward model
    iter_dict['Radon_algo'] = 'ind'
    #iter_dict['Radon_algo'] = 'astra'
    iter_dict['interpolation'] = 1 # 0 for nearest neighbor, 1 for linear
    iter_dict['gamma_abs'] = 1 # use absorption data?
    iter_dict['gamma_phs'] = 1 # use phase contrast data?
    
    # noise suppression parameters
    iter_dict['lambda1'] = 0 # for l2 norm should be at least 10*mean of rec
    iter_dict['delta1'] = 100
    iter_dict['lambda2'] = 0
    iter_dict['delta2'] = 100
    
    # parameters for the radon lookup table
    iter_dict['RA'] = 128 # this is needed if gradradon_x are not given
    iter_dict['start'] = 0 # not used
    iter_dict['theta'] = 0 # rotation angles
    iter_dict['weights'] = 1 #  leave at 1
    iter_dict['gradradon_inds'] = 0
    iter_dict['gradradon_vals'] = 0
    iter_dict['graddiffradon_vals'] = 0
    iter_dict['graddiffradon_inds'] = 0
    
    
    return iter_dict

def gen_rec_dict():
    """
    USAGE:
        gen_iter_dict = iterlib.gen_rec_dict()
        
    DESCRIPTION:
        returns an empty dictionary of iteration results
    
    """
    rec_dict = {}
    rec_dict['Nrecvalues'] = []
    rec_dict['BFGS_results'] = []
    rec_dict['rec_slice_abs'] = []
    rec_dict['rec_slice_abs_inds'] = []
    rec_dict['rec_slice_phs'] = []
    rec_dict['rec_slice_phs_inds'] = []
    rec_dict['rec_mo'] = []
    rec_dict['rec_mo_inds'] = []
    rec_dict['rec_gamma'] = []
    rec_dict['rec_gamma_inds'] = []
    rec_dict['rec_icfac'] = []
    rec_dict['rec_icfac_inds'] = []
    rec_dict['rec_mr'] = []
    rec_dict['rec_mr_inds'] = []
    rec_dict['rec_flatm'] = []
    rec_dict['rec_flatm_inds'] = []
    rec_dict['Liter'] = []
    return rec_dict

def perc(I, p = 99):
    "Usage: I = pm.perc(I, p = 95); percentile an image"
    zhih = np.percentile(I, p)
    zlow = np.percentile(I, 100-p)
    II = I.copy()
    II[I>zhih] = zhih
    II[I<zlow] = zlow
    return II

def radonpicker(_image,_theta,algo='astra',*argv):
    """
    USAGE: sino = radonpicker(org,theta,algo,args)
    
        algo='ind': Radon_algo_args = gradradon(_image.shape[0],theta,RA)
                    radonpicker(org,theta,'ind', Radon_algo_args)
                    
                    RA is in the definition of gradradon_inds & gradradon_vals
    """
    
#    print(argv)    
    if algo == 'astra':
        if not argv:
            method = 'cuda'
        else:
            method = argv[0]
        sino = pm.radonastra(_image,_theta,'cuda') # not used here
    elif algo == 'fft1':
        sino = pm.radonfft1(_image,_theta) # not used here
    elif algo == 'ski':
        if not argv:
            cir = True
        else:
            cir = argv
        sino = radon(_image,_theta,circle=cir)
    elif algo == 'ind':
        if not argv:
            print("setting up radon lookup table")
            RA = _image.shape[0]//2
            indsl, valsl = gradradon(_image.shape[0],_theta,RA)
        else:
            indsl, valsl = argv[0][0], argv[0][1]
        sino = radonind(_image,_theta,indsl,valsl)
    else:
        print("Unkown algo")
    return sino

def gradradon(Ndet,theta,RA,inter=1):
    """Usage:
        
        gradradon_inds, gradradon_vals = gradradon(Ndet,theta,RA,inter=1)
        
        inter = 1: use interpolation
        inter = 0: use nearest neighbour
        
    """
    
    gradradon_inds = np.asarray([None] *Ndet*Ndet) # indices
    gradradon_vals = np.asarray([None] *Ndet*Ndet) # values
    Ntheta = len(theta)
    Ntrange = np.arange(Ntheta)
    n = 1.5 # this was found empirically, matching astra radon the best


    for isi in range(Ndet*Ndet):
        isy, isx = np.unravel_index(isi,(Ndet,Ndet))
        xs = isx-Ndet/2+0.5
        ys = isy-RA
        r = np.sqrt(xs**2+ys**2)
        phi = np.arctan2(ys,xs)
        t = r*np.cos(np.deg2rad(theta)+phi)+RA
        if not inter:
            t = np.round(t).astype('int32')
            if np.prod(t>0)*np.prod(t+1<Ndet):     
                gradradon_inds[isi] = np.ravel_multi_index((t,Ntrange),(Ndet,Ntheta))
                gradradon_vals[isi] = np.ones(Ntheta)
            else:
                gradradon_inds[isi] = []
                gradradon_vals[isi] = []
        
        else: 
            finds = np.floor(t).astype('int32') # for integers floor & ceil are the same!
            cinds = np.ceil(t).astype('int32') # uint32 implies a limitation of sinogram size to ~65000x65000
    #        if np.prod(cinds < Ndet)*np.prod(finds >0): # check if x,y is within reconstruction circle
            if np.prod(cinds+1 < Ndet)*np.prod(finds > 0): # this must be the same constrains as with graddiffradon
    
                d1 = t-finds
                d2 = cinds-t
        
                iinds = np.where(finds==cinds)[0]
                ninds = np.where(finds!=cinds)[0]
    
                a = 1/(d1[ninds]**n+d2[ninds]**n)
    
                arr1 = np.ravel_multi_index((finds[iinds],Ntrange[iinds]),(Ndet,Ntheta))
                arr2 = np.ones(len(iinds))
                arr1 = np.append(arr1,np.ravel_multi_index((finds[ninds],Ntrange[ninds]),(Ndet,Ntheta)))
                arr2 = np.append(arr2,a*d2[ninds]**n)
                arr1 = np.append(arr1,np.ravel_multi_index((cinds[ninds],Ntrange[ninds]),(Ndet,Ntheta)))
                arr2 = np.append(arr2,a*d1[ninds]**n)
                      
                aa = np.argsort(arr1)
                
                gradradon_inds[isi] = np.uint32(arr1[aa])
                gradradon_vals[isi] = np.float32(arr2[aa])
        
            else:
                gradradon_inds[isi] = []
                gradradon_vals[isi] = []
        
    return gradradon_inds, gradradon_vals


def graddiffradon(Ndet,theta,RA,inter=1):
    """Usage: 
        
        graddiffradon_inds, graddiffradon_vals = graddiffradon(Ndet,theta,RA,inter=1)
                
        inter = 1: use interpolation
        inter = 0: use nearest neighbour
        
    """
#    print('hh')
    graddiffradon_inds = np.asarray([None] *Ndet*Ndet) # indices
    graddiffradon_vals = np.asarray([None] *Ndet*Ndet) # values
    Ntheta = len(theta)
    Ntrange = np.arange(Ntheta)
    n = 1.5 # this was found empirically, matching astra radon the best
    
    for isi in range(Ndet*Ndet):

        isy, isx = np.unravel_index(isi,(Ndet,Ndet))
        xs = isx-Ndet/2+0.5
        ys = isy-RA
        r = np.sqrt(xs**2+ys**2)
        phi = np.arctan2(ys,xs)
        t = r*np.cos(np.deg2rad(theta)+phi)+RA#+5.4e-15
        
        if not inter:
            t = np.round(t).astype('int32')
            if np.prod(t>0)*np.prod(t+1<Ndet):
                arr_inds = np.ravel_multi_index((t,Ntrange),(Ndet,Ntheta))
                arr_vals = np.ones(len(t))
                arr_inds = np.append(arr_inds, np.ravel_multi_index((t+1,Ntrange),(Ndet,Ntheta)))
                arr_vals = np.append(arr_vals,-np.ones(len(t)))
                aa = np.argsort(arr_inds)
                                
                graddiffradon_inds[isi] = np.uint32(arr_inds[aa])
                graddiffradon_vals[isi] = np.float32(arr_vals[aa])
                
            else:
                graddiffradon_inds[isi] = []
                graddiffradon_vals[isi] = []
        else:
            
            finds = np.floor(t).astype('int32') # for integerss floor & ceil are the same!
            cinds = np.ceil(t).astype('int32')
    #        print(t)
            if np.prod(cinds+1 < Ndet)*np.prod(finds > 0): # check if x,y is within reconstruction circle
        
                d1 = t-finds
                d2 = cinds-t
        
                iinds = np.where(finds==cinds)[0]
                ninds = np.where(finds!=cinds)[0]
                a = 1/(d1[ninds]**n+d2[ninds]**n) # something is wrong here
                
                arr1 = np.ravel_multi_index((finds[iinds],Ntrange[iinds]),(Ndet,Ntheta))
                arr1 = np.append(arr1,np.ravel_multi_index((finds[iinds]+1,Ntrange[iinds]),(Ndet,Ntheta)))
                
                arr1 = np.append(arr1,np.ravel_multi_index((finds[ninds],Ntrange[ninds]),(Ndet,Ntheta)))
                arr1 = np.append(arr1,np.ravel_multi_index((cinds[ninds],Ntrange[ninds]),(Ndet,Ntheta)))
                arr1 = np.append(arr1,np.ravel_multi_index((cinds[ninds]+1,Ntrange[ninds]),(Ndet,Ntheta)))
                
                arr2 = np.ones(len(iinds))
                arr2 = np.append(arr2,-np.ones(len(iinds)))
                
                arr2 = np.append(arr2,a*d2[ninds]**n)
                arr2 = np.append(arr2,-a*d2[ninds]**n+a*d1[ninds]**n)
                arr2 = np.append(arr2,-a*d1[ninds]**n)
                      
                aa = np.argsort(arr1)
                
                graddiffradon_inds[isi] = np.uint32(arr1[aa])
                graddiffradon_vals[isi] = np.float32(arr2[aa])
        
            else:
                graddiffradon_inds[isi] = []
                graddiffradon_vals[isi] = []
                
    return graddiffradon_inds, graddiffradon_vals

def radonind(_image,_theta,indsl,valsl):
    sino = np.zeros((_image.shape[0],len(_theta))).flatten()
    zorg = _image.flatten()
    for isi in range(_image.shape[0]*_image.shape[0]):
        if len(indsl[isi]):
            sino[indsl[isi]] += valsl[isi]*zorg[isi]
    return np.reshape(sino,(_image.shape[0],len(_theta)))

def single_shot(x0,xs,ICflat,samp,theta,gamma,dorec=1):
    """
    USAGE: result = single_shot(x0,xs,ICflat,samp,theta,gamma,dorec=1)
    
        x0: working point on the IC
        xs: scan positions for ICflat
        ICflat: flat IC - ICflat[Nvoxel,len(xs)]
        samp: sample images - samp[Nvoxel,Ntheta]
        theta: rotation angles
        gamma: gamma factor
        dorec = 1: do tomographic rec or just return single shot results
        
        Assumes distance z=1
    """
    
    Nvoxel = samp.shape[0]
    Ntheta = len(theta)
    Dxs = xs[1]-xs[0]
    xvoxel = np.linspace(0,Nvoxel,Nvoxel,endpoint=False)

    f0 = np.zeros((Nvoxel))
    fs = np.zeros((Nvoxel))   
    for isp in range(Nvoxel):
        f0[isp] = np.interp(x0,xs,ICflat[isp,:])
    fs =  np.interp(x0,xs,np.append(0,np.diff(np.mean(ICflat,0))))/Dxs

    c = fs/f0 
    
    mix_sino = np.zeros((Nvoxel,Ntheta))
    qfr = np.fft.fftfreq(Nvoxel,d=xvoxel[1]-xvoxel[0])
    for ist in range(len(theta)):
                
        hsf = np.fft.fft(np.fft.fftshift(samp[:,ist]/f0))
        hhsf = hsf/(1-1j*gamma*c*qfr*np.pi*2)
        rho = -gamma*np.log(np.real(np.fft.fftshift(np.fft.ifft(hhsf))))
        mix_sino[:,ist] = rho#-rho.min()

    if dorec: # do tomo rec here
        result = iradon(mix_sino,theta,circle=True)
    else:
        result = mix_sino
        
    return result


# iter_tomo_rec is a test function for plain reconstruction. Please ignore
def iter_tomo_rec(sino,iter_dict):
    
    def L(p):
        pradon = radonpicker(np.reshape(p,((Ndet,Ndet))),theta,Radon_algo,Radon_algo_args)
        return np.sum(weights*(sino-pradon)**2)/C + lambda1*Huber(p)
    
    def gradL(p):
        pradon = radonpicker(np.reshape(p,((Ndet,Ndet))),theta,Radon_algo,Radon_algo_args)
        
        temp = -2*(weights*(sino-pradon)).flatten()/C
        gradLvals = np.zeros(Ndet*Ndet)
        
        for isi in range(Ndet*Ndet):
            if len(gradradon_inds[isi]):
                gradLvals[isi] = np.sum(temp[gradradon_inds[isi]]*gradradon_vals[isi])
                
        if lambda1: # using Huber regularisation
            gradLvals += lambda1*gradHuber(p)
            
        return gradLvals
    
    def Huber(p):
        L = np.sum(delta1**2*(np.sqrt(1+(p/delta1)**2)-1))
        return L
    
    def gradHuber(p):
        gradL = delta1*p/np.sqrt(delta1**2+p**2)
        return gradL
    
    def gradLnum(p):
        # numerical one could be off due to rounding errors
        # this is true if L(p-ff*alpha) > L(p-fc*alpha) for some small alpha
        # This means that the analytical grad is better
        gradLnum_ = np.zeros(len(p))
        for isn in range(len(p)):
            p0 = p.copy(); p0[isn] = p[isn]+1e-7
            gradLnum_[isn] = (L(p0)-L(p))/1e-7
        return gradLnum_
    
    def gradHubernum(p):
        gradHnum_ = np.zeros(len(p))
        for isn in range(len(p)):
            p0 = p.copy(); p0[isn] = p[isn]+1e-7
            gradHnum_[isn] = (Huber(p0)-Huber(p))/1e-7
        return gradHnum_
    
    def checkgrad(p):
        from scipy import stats
    
        ff = gradLnum(p);
        fc = gradL(p);
        
        pl.subplot(221)
        pl.plot(ff)
        pl.plot(fc)
        pl.subplot(222)
        pl.plot(ff,fc,'.')
        pl.subplot(223)
        pl.imshow(ff.reshape((Ndet,Ndet)));pl.colorbar();
        pl.title('num')
        pl.subplot(224)
        pl.imshow(fc.reshape((Ndet,Ndet)));pl.colorbar()
        pl.title('ana')
        pl.show()
        print('grad for rec slice:')
        print('   ff mean: ', np.mean(ff))
        print('   fc mean: ', np.mean(fc))
        slope, intercept, r_value, p_value, std_err = stats.linregress(ff,fc)
        print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value) 
    
    def callback(p):
        pl.imshow(perc(np.reshape(p,((Ndet,Ndet)))),vmin=0)
        pl.show()
        print(L(p))
        
    
    # routine starts here
    Ndet = sino.shape[0]
    theta = iter_dict['theta']
    Ntheta = len(theta)
    RA = iter_dict['RA']    
    Radon_algo = iter_dict['Radon_algo']  
    lambda1 = iter_dict['lambda1']
    delta1 = iter_dict['delta1']
    weights = iter_dict['weights']
    start = iter_dict['start']
    if len(iter_dict['gradradon_inds']):
        gradradon_inds = iter_dict['gradradon_inds']
        gradradon_vals = iter_dict['gradradon_vals']
    else:
        print('generating radon lookup')
        gradradon_inds, gradradon_vals = gradradon(Ndet,theta,RA)
    Radon_algo_args = gradradon_inds, gradradon_vals
    C = np.sum(sino)
    Nrecvalues = Ndet*Ndet
    
    if iter_dict['callback']:
        cbfun = callback
    else:
        cbfun = None

    
    #%% starting values
    if not np.any(start): # start value was given as 0    
        start = np.zeros(Nrecvalues).flatten()    
    else:
        start = start
    
    #%% bounds
    lbounds = np.zeros((Ndet*Ndet,1))
    ubounds = np.inf*np.ones((Ndet*Ndet,1))
    bounds = np.hstack((lbounds,ubounds))
    
    #%% minimisation
    tt = time.time()
    res3 = optimize.fmin_l_bfgs_b(L, start, bounds=bounds, fprime=gradL,
                                  m=iter_dict['m'], factr=iter_dict['factr'], pgtol=1e-10, epsilon=1e-08, iprint=1,
                                  maxfun=15000, maxiter=iter_dict['maxiter'], disp=True, callback=cbfun,
                                  maxls=iter_dict['maxls'])
    
    print(res3[2]['task'])
    print('iterations done: ', res3[2]['nit'])
    print("time list-line CG: ",time.time()-tt)
    
    rec = np.reshape(res3[0],((Ndet,Ndet)))
    funcs = [L,gradL,checkgrad,gradHuber,gradHubernum] # to allow access to the functions outside   
    
    print("S(rec): ", L(res3[0]))
    print("Huber(rec): ", lambda1*Huber(rec.flatten()))
    print('mean rec: ', np.mean(np.abs(rec)))
    
    return rec, funcs


#@profile
def iter_single_shot(x,f,m,s,iter_dict):
    """
    USAGE:
         rec_dict, funcs = iterlib.iter_single_shot(mflat,ICflat,m0,ICsamp,iter_dict)
         
    It says single shot, but separated reconstruction of absorption and phase is implemented
    """
    
    global Niter, slice_abs, mo, mr, flatm, Liter, gamma, icfac
    Niter = 0
    slice_abs = iter_dict['slice_abs']
    mo = iter_dict['mo']
    mr = iter_dict['mr']
    flatm = iter_dict['flatm']
    gamma = iter_dict['gamma']
    icfac = iter_dict['icfac']
    Liter = []
    
    def L(p):
        if iter_dict['do_rec_slice_abs']:
            if iter_dict['do_rec_slice_phs']:
                return S(p)+lambda1*Huber(p[rec_slice_abs_inds],delta1)+lambda2*Huber(p[rec_slice_phs_inds],delta2)
            else: # single shot
                return S(p)+lambda1*Huber(p[rec_slice_abs_inds],delta1)
        else:
            return S(p)
            
    def gradL(p):
        if iter_dict['do_rec_slice_abs']:
            if iter_dict['do_rec_slice_phs']:
                return  gradS_list_line2(p) \
                        + np.append(np.append(lambda1*gradHuber(p[rec_slice_abs_inds],delta1 ), \
                        lambda2*gradHuber(p[rec_slice_phs_inds],delta2)),np.zeros(Nrecvalues-2*len(rec_slice_abs_inds)))
            else:
                return gradS_list_line2(p)\
                        + np.append(lambda1*gradHuber(p[rec_slice_abs_inds],delta1),np.zeros(Nrecvalues-len(rec_slice_abs_inds)))
        else:
            return gradS_list_line2(p)
    
#    @profile
    def S(p):
        global slice_abs, slice_phs, mo , mr, flatm, gamma, icfac
        
        if iter_dict['do_rec_slice_abs']:
            slice_abs = p[rec_slice_abs_inds]
        if iter_dict['do_rec_slice_phs']:
            slice_phs = p[rec_slice_phs_inds]
        if iter_dict['do_rec_mo']:
            mo = np.reshape(p[rec_mo_inds],(Ntheta,M))
        if iter_dict['do_rec_icfac']:
            icfac = p[rec_icfac_inds]
        if iter_dict['do_rec_gamma']:
            gamma = p[rec_gamma_inds]
        if iter_dict['do_rec_mr']:
#            mr = np.append(p[rec_mr_inds],0)
            mr = p[rec_mr_inds]
        if iter_dict['do_rec_flatm']:
            flatm = p[rec_flatm_inds]
#        print('icfac: ', icfac)
#        print('gamma: ', gamma)
        
        if iter_dict['do_rec_slice_abs']:
            fradon = radonpicker(np.reshape(slice_abs,((Ndet,Ndet))),theta,Radon_algo,Radon_algo_args)    
            Aradon = np.exp(-gamma_abs*fradon)
        else:
            Aradon = np.ones((Ndet,Ntheta))
            
        if iter_dict['do_rec_slice_phs']:
            Dradon = np.zeros((Ndet,Ntheta))
#            Dradon = gamma_phs*radonpicker(np.reshape(slice_phs,((Ndet,Ndet))),theta,Radon_algo,Radon_algo_args2)
            Dradon[1:,:] = gamma_phs*np.diff(radonpicker(np.reshape(slice_phs,((Ndet,Ndet))),theta,Radon_algo,Radon_algo_args),1,0)
        else: # single shot
            Dradon = np.zeros((Ndet,Ntheta))    
            Dradon[1:,:] =  gamma*gamma_phs*np.diff(fradon,1,0)
            
        fs = np.zeros((Ndet,Ntheta,M))
#        print(mo.shape)
#        print(mr.shape)
#        print(np.matlib.repmat(Dradon[0,:],M,1).transpose().shape)
#        print((fs[0,:,:]).shape())
        for isp in range(Ndet):
            fs[isp,:,:] = np.matlib.repmat(Aradon[isp,:],M,1).transpose()*\
                        np.interp(m-mo-mr[isp]+np.matlib.repmat(Dradon[isp,:],M,1).transpose(),x+flatm,f[isp,:]*icfac)
        
        return np.sum(weight*(s-fs) **2)/C

#    @profile    
    def gradS_list_line2(p):    
        global slice_abs, slice_phs, mo, mr, flatm, gamma, icfac
        
        if iter_dict['do_rec_slice_abs']:
            slice_abs = p[rec_slice_abs_inds]
        if iter_dict['do_rec_slice_phs']:
            slice_phs = p[rec_slice_phs_inds]
        if iter_dict['do_rec_mo']:
            mo = np.reshape(p[rec_mo_inds],(Ntheta,M))
        if iter_dict['do_rec_icfac']:
            icfac = p[rec_icfac_inds]
        if iter_dict['do_rec_gamma']:
            gamma = p[rec_gamma_inds]     
        if iter_dict['do_rec_mr']:
#            mr = np.append(p[rec_mr_inds],0)
            mr = p[rec_mr_inds]
        if iter_dict['do_rec_flatm']:
            flatm = p[rec_flatm_inds]
            
#        print('icfac: ', icfac)
#        print('gamma: ', gamrec_)

        
        if iter_dict['do_rec_slice_abs']:
            fradon = radonpicker(np.reshape(slice_abs,((Ndet,Ndet))),theta,Radon_algo,Radon_algo_args)    
            Aradon = np.exp(-gamma_abs*fradon)
        else:
            Aradon = np.ones((Ndet,Ntheta))
        if iter_dict['do_rec_slice_phs']:
            Dradon = np.zeros((Ndet,Ntheta))
#            Dradon = gamma_phs*radonpicker(np.reshape(slice_phs,((Ndet,Ndet))),theta,Radon_algo,Radon_algo_args2)
            Dradon[1:,:] = gamma_phs*np.diff(radonpicker(np.reshape(slice_phs,((Ndet,Ndet))),theta,Radon_algo,Radon_algo_args),1,0)
        else:
            Dradon = np.zeros((Ndet,Ntheta))    
            Dradon[1:,:] =  gamma*gamma_phs*np.diff(fradon,1,0)
            
        fs = np.zeros((Ndet,Ntheta,M))
        fss = np.zeros((Ndet,Ntheta,M))
        ff1 = np.zeros((Ndet,Ntheta,M))
            
        for isp in range(Ndet):
            AAradon = np.matlib.repmat(Aradon[isp,:],M,1).transpose()
            DDradon = np.matlib.repmat(Dradon[isp,:],M,1).transpose()
            fs[isp,:,:]  = AAradon*np.interp(m-mo-mr[isp]+DDradon,x+flatm,f[isp,:]*icfac)
            fss[isp,:,:] = AAradon*np.interp(m+Dx/2-mo-mr[isp]+DDradon,x[1:]+flatm[1:],np.diff(f[isp,:])*icfac/Dx)
#            fss[isp,:,:] = AAradon*np.interp(m-mo-mr[isp]+DDradon,x[1:],np.diff(f[isp,:])*icfac/Dx)
            ff1[isp,:,:]  = AAradon*np.interp(m-mo-mr[isp]+DDradon,x+flatm,f[isp,:])    
        
        gradS_list_line = np.zeros(Nrecvalues)   
        if iter_dict['do_rec_slice_abs']:
             
            temp1 =  2*np.sum((weight*(s-fs))*fs,2).flatten()*gamma_abs/C
            temp2 =  2*np.sum((weight*(fs-s))*fss,2).flatten()*gamma*gamma_phs/C
            
            if iter_dict['do_rec_slice_phs']: # do sep. rec
                for isi in range(Ndet*Ndet):
                    if len(gradradon_inds[isi]):
                        gradS_list_line[isi] = np.sum(temp1[gradradon_inds[isi]]*gradradon_vals[isi]) 
                        gradS_list_line[isi+Ndet*Ndet] = np.sum(temp2[graddiffradon_inds[isi]]*graddiffradon_vals[isi])      
            else: # single shot
                for isi in range(Ndet*Ndet):
                    if len(gradradon_inds[isi]):
                        gradS_list_line[isi] = np.sum(temp1[gradradon_inds[isi]]*gradradon_vals[isi]) \
                                             + np.sum(temp2[graddiffradon_inds[isi]]*graddiffradon_vals[isi])
        
        elif iter_dict['do_rec_slice_phs']:
            
#            temp1 =  2*np.sum((weight*(s-fs))*fs,2).flatten()*gamma_abs/C
            temp2 =  2*np.sum((weight*(fs-s))*fss,2).flatten()*gamma*gamma_phs/C
            
            for isi in range(Ndet*Ndet):
                    if len(gradradon_inds[isi]):
                        gradS_list_line[isi] = np.sum(temp2[graddiffradon_inds[isi]]*graddiffradon_vals[isi])      
            
        
        if iter_dict['do_rec_mo']:
            # abnormal termination has nothing to do with how this grad is computed
            # it seems to be an effect of noise in measurements
            # offset between gradana & gradnum seems to be due to noise
#            gradS_list_line[rec_mo_inds] = 2/C*(np.sum(weight*s*fss,0)-np.sum(weight*fs*fss,0)).flatten()
            gradS_list_line[rec_mo_inds] = 2/C*(np.sum(weight*(s-fs)*fss,0)).flatten()
#            for isr in range(len(rec_mo_inds)):
#                p0 = p.copy(); p0[rec_mo_inds[isr]] = p[rec_mo_inds[isr]]+1e-11
#                gradS_list_line[rec_mo_inds[isr]] = (L(p0)-L(p))/1e-11
#            
        if iter_dict['do_rec_icfac']:
            gradS_list_line[rec_icfac_inds] = -2*np.sum(weight*(s-fs)*ff1)/C   
        if iter_dict['do_rec_gamma']:
            gradS_list_line[rec_gamma_inds] = -2*np.sum(np.squeeze(weight*(s-fs)*fss)*Dradon/gamma).flatten()/C
        if iter_dict['do_rec_mr']:
#            gradS_list_line[rec_mr_inds] = (2*np.sum(weight*(s-fs)*fss,1).flatten()/C)[:-1]
            gradS_list_line[rec_mr_inds] = 2/C*np.sum(weight*(s-fs)*fss,1).flatten()
        if iter_dict['do_rec_flatm']:
            LL = S(p)
            for ism in range(Mflat):
                p0 = p.copy();
                p0[rec_flatm_inds[ism]] = p0[rec_flatm_inds[ism]]+1e-8
                gradS_list_line[rec_flatm_inds[ism]] = (S(p0)-LL)/1e-8
            
        return gradS_list_line
    
    def Huber(p,delta):
        # gradS & gradHuber will have opposite directions
        # -> too high lambda's will lead to total gradL ~0
        # -> long or futile iteration
        return delta**2*np.sum(np.sqrt(1+(p/delta)**2)-1)
    
    def gradHuber(p,delta):
        return delta*p/np.sqrt(delta**2+p**2)
    
    def gradSnum(p):
        gradSnum_ = np.zeros(len(p))
        S_ = S(p)
        for isn in range(len(p)):
            p0 = p.copy(); p0[isn] = p[isn]+1e-7
            gradSnum_[isn] = (S(p0)-S_)/1e-7
            
        return gradSnum_
    
    def gradLnum(p):
        # numerical one could be off due to rounding errors
        # this is true if L(p-ff*alpha) > L(p-fc*alpha) for some small alpha
        # This means that the analytical grad is better
        gradLnum_ = np.zeros(len(p))
        L_ = L(p)
        for isn in range(len(p)):
            p0 = p.copy(); p0[isn] = p[isn]+1e-7
            gradLnum_[isn] = (L(p0)-L_)/1e-7
            
        return gradLnum_
    
    def gradHubernum(p,delta):
        gradHnum_ = np.zeros(len(p))
        for isn in range(len(p)):
            p0 = p.copy(); p0[isn] = p[isn]+1e-7
            gradHnum_[isn] = (Huber(p0,delta)-Huber(p,delta))/1e-7
            
        return gradHnum_
    
    def callback(p):
        global Niter, Liter
        if iter_dict['do_rec_mo'] and iter_dict['do_rec_mr']:      
            pl.subplot2grid((2,3), (0,0), colspan=2,rowspan=2)
            pl.imshow(perc(np.reshape(p[rec_slice_abs_inds],(Ndet,Ndet))));pl.colorbar();
            pl.title('Niter: '+  n2s(Niter,5))
            ax=pl.subplot2grid((2,3), (0,2))
            pl.plot(p[rec_mo_inds])
            pl.title('mo');pl.ylim((-.3,.3))
            ax.yaxis.tick_right()            
            ax.yaxis.set_label_position("right")
            ax=pl.subplot2grid((2,3), (1,2))
            pl.plot(p[rec_mr_inds])
            pl.title('mr');pl.ylim((-.2,.2))
            ax.yaxis.tick_right()            
            ax.yaxis.set_label_position("right")
        elif iter_dict['do_rec_mo']:
#            pl.subplot(1,3,(1,2))
#            pl.imshow(perc(np.reshape(p[rec_slice_abs_inds],(Ndet,Ndet))));pl.colorbar();
            pl.subplot(133)
            pl.plot(p[rec_mo_inds])
        elif iter_dict['do_rec_mr']:
            pl.subplot(1,3,(1,2))
            pl.imshow(perc(np.reshape(p[rec_slice_abs_inds],(Ndet,Ndet))));pl.colorbar();
            pl.subplot(133)
            pl.plot(p[rec_mr_inds])
        else:
            pl.imshow(perc(np.reshape(p[rec_slice_abs_inds],(Ndet,Ndet))));pl.colorbar();
        
        if iter_dict['do_rec_slice_phs']:
            pl.show()
            pl.imshow(perc(np.reshape(-p[rec_slice_phs_inds],(Ndet,Ndet))));pl.colorbar(); pl.show()
            print('')
        
        Liter = np.append(Liter,L(p))
        print(Liter[Niter])
        if iter_dict['do_rec_icfac']:
            print('icfac: ', p[rec_icfac_inds])
        if iter_dict['do_rec_gamma']:
            print('gamma: ', p[rec_gamma_inds])
        
        
        
#        pl.savefig('C:\\Users\\Peter\\Documents\\Documents\\EI\\eval\\perkin_elmer\\iter\\presentation\\gif\\osa_' + n2s(Niter,5) + '.png')
        
        pl.show()
        Niter += 1
    
    # check gradient
    def checkgrad(p):
    # iinds != 0 could be removed because reconstruction circles were fixed
        
    
#        p = np.zeros(Nrecvalues).flatten()

        ff = gradLnum(p);
        fc = gradL(p);
        
        if iter_dict['do_rec_slice_abs']:                        
            iinds = rec_slice_abs_inds
            print('grad for rec abs slice L(p):')
            print('   fnum norm: ', np.linalg.norm(ff[iinds]))
            print('   fana mean: ', np.linalg.norm(fc[iinds]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(ff[iinds],fc[iinds])
            print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value)
            pl.subplot(221)
            pl.plot(ff[iinds]);
            pl.plot(fc[iinds])
            pl.subplot(222)
            pl.plot(ff[iinds],fc[iinds],'.')
            pl.subplot(223)
            pl.imshow(ff[iinds].reshape((Ndet,Ndet)));pl.colorbar();
            pl.title('num')
            pl.subplot(224)
            pl.imshow(fc[iinds].reshape((Ndet,Ndet)));pl.colorbar()
            pl.title('ana')
            pl.show()
            
            ffs = gradSnum(p)
            fcs = gradS_list_line2(p)
            print('grad for rec abs slice S(p):')
            print('   fnum mean: ', np.mean(ffs[iinds]))
            print('   fana mean: ', np.mean(fcs[iinds]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(ffs[iinds],fcs[iinds])
            print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value)
            pl.subplot(221)
            pl.plot(ffs[iinds]);
            pl.plot(fcs[iinds])
            pl.subplot(222)
            pl.plot(ffs[iinds],fcs[iinds],'.')
            pl.subplot(223)
            pl.imshow(ffs[iinds].reshape((Ndet,Ndet)));pl.colorbar();
            pl.title('num')
            pl.subplot(224)
            pl.imshow(fcs[iinds].reshape((Ndet,Ndet)));pl.colorbar()
            pl.title('ana')
            pl.show()
            
            if iter_dict['lambda1']:
                Hana1 = gradHuber(p[rec_slice_abs_inds],delta1)       
                Hnum1 = gradHubernum(p[rec_slice_abs_inds],delta1)
   
                print('grad for rec abs slice Huber1(p):')
                print('   Hnum1 mean: ', np.mean(Hnum1))
                print('   Hana1 mean: ', np.mean(Hana1))
                slope, intercept, r_value, p_value, std_err = stats.linregress(Hnum1,Hana1)
                print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value)
                pl.subplot(221)
                pl.plot(Hnum1);
                pl.plot(Hana1)
                pl.subplot(222)
                pl.plot(Hnum1,Hana1,'.')
                pl.subplot(223)
                pl.imshow(Hnum1.reshape((Ndet,Ndet)));pl.colorbar();
                pl.title('num')
                pl.subplot(224)
                pl.imshow(Hana1.reshape((Ndet,Ndet)));pl.colorbar()
                pl.title('ana')
                pl.show()             
                    
        
        if iter_dict['do_rec_slice_phs']:                        
            iinds = rec_slice_phs_inds
            print('grad for rec phs slice L(p):')
            print('   ff mean: ', np.mean(ff[iinds]))
            print('   fc mean: ', np.mean(fc[iinds]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(ff[iinds],fc[iinds])
            print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value)
            pl.subplot(221)
            pl.plot(ff[iinds]);
            pl.plot(fc[iinds])
            pl.subplot(222)
            pl.plot(ff[iinds],fc[iinds],'.')
            pl.subplot(223)
            pl.imshow(ff[iinds].reshape((Ndet,Ndet)));pl.colorbar();
            pl.title('num')
            pl.subplot(224)
            pl.imshow(fc[iinds].reshape((Ndet,Ndet)));pl.colorbar()
            pl.title('ana')
            pl.show()
            
            ffs = gradSnum(p)
            fcs = gradS_list_line2(p)
            print('grad for rec phs slice S(p):')
            print('   ff mean: ', np.mean(ffs[iinds]))
            print('   fc mean: ', np.mean(fcs[iinds]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(ffs[iinds],fcs[iinds])
            print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value)
            pl.subplot(221)
            pl.plot(ffs[iinds]);
            pl.plot(fcs[iinds])
            pl.subplot(222)
            pl.plot(ffs[iinds],fcs[iinds],'.')
            pl.subplot(223)
            pl.imshow(ffs[iinds].reshape((Ndet,Ndet)));pl.colorbar();
            pl.title('num')
            pl.subplot(224)
            pl.imshow(fcs[iinds].reshape((Ndet,Ndet)));pl.colorbar()
            pl.title('ana')
            pl.show()
        
            if iter_dict['lambda1'] > 0 and np.sum(np.abs(p))>0:
                
                ffh = iter_dict['lambda1']*gradHubernum(p[rec_slice_abs_inds],iter_dict['delta1'])
                fch = iter_dict['lambda1']*gradHuber(p[rec_slice_abs_inds],iter_dict['delta1'])
                iinds = rec_slice_abs_inds
                print('grad for lambda1*Huber abs slice:')
                print('   ff mean: ', np.mean(ffh[iinds]))
                print('   fc mean: ', np.mean(fch[iinds]))
                slope, intercept, r_value, p_value, std_err = stats.linregress(ffh[iinds],fch[iinds])
                print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value)
                pl.subplot(221)
                pl.plot(ffh[iinds]);
                pl.plot(fch[iinds])
                pl.subplot(222)
                pl.plot(ffh[iinds],fch[iinds],'.')
                pl.subplot(223)
                pl.imshow(ffh[rec_slice_abs_inds].reshape((Ndet,Ndet)));pl.colorbar();
                pl.title('num')
                pl.subplot(224)
                pl.imshow(fch[rec_slice_abs_inds].reshape((Ndet,Ndet)));pl.colorbar()
                pl.title('ana')
                pl.show()
                
        
        if iter_dict['lambda2'] > 0 and np.sum(np.abs(p))>0:
                
                ffh = iter_dict['lambda2']*gradHubernum(p[rec_slice_phs_inds],iter_dict['delta2'])
                fch = iter_dict['lambda2']*gradHuber(p[rec_slice_phs_inds],iter_dict['delta2'])
                iinds = rec_slice_phs_inds
                print('grad for lambda2*Huber phs slice:')
                print('   ff mean: ', np.mean(ffh))
                print('   fc mean: ', np.mean(fch))
                slope, intercept, r_value, p_value, std_err = stats.linregress(ffh,fch)
                print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value)
                pl.subplot(221)
                pl.plot(ffh);
                pl.plot(fch)
                pl.subplot(222)
                pl.plot(ffh,fch,'.')
                pl.subplot(223)
                pl.imshow(ffh.reshape((Ndet,Ndet)));pl.colorbar();
                pl.title('num')
                pl.subplot(224)
                pl.imshow(fch.reshape((Ndet,Ndet)));pl.colorbar()
                pl.title('ana')
                pl.show()
        
        if iter_dict['do_rec_mo']:
            print('grad for rec mo:')
            print('   ff mean: ', np.mean(ff[rec_mo_inds]))
            print('   fc mean: ', np.mean(fc[rec_mo_inds]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(ff[rec_mo_inds],fc[rec_mo_inds])
            print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value)
            pl.subplot(221)
            pl.plot(ff[rec_mo_inds]);
            pl.plot(fc[rec_mo_inds])
            pl.subplot(222)
            pl.plot(ff[rec_mo_inds],fc[rec_mo_inds],'.')
            pl.show()
            
        if iter_dict['do_rec_icfac']:
            print('grad for rec icfac:')
            print('   icfac   : ', p[rec_icfac_inds])
            print('   gradLnum: ', ff[rec_icfac_inds])
            print('   gradLana: ', fc[rec_icfac_inds])
            
        if iter_dict['do_rec_gamma']:
            print('grad for rec gamma:')
            print('   gamma   : ', p[rec_gamma_inds])
            print('   gradLnum: ', ff[rec_gamma_inds])
            print('   gradLana: ', fc[rec_gamma_inds])
            
        if iter_dict['do_rec_mr']:
            print('grad for rec mr:')
            print('   ff mean: ', np.mean(ff[rec_mr_inds]))
            print('   fc mean: ', np.mean(fc[rec_mr_inds]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(ff[rec_mr_inds],fc[rec_mr_inds])
            print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value)
            pl.subplot(221)
            pl.plot(ff[rec_mr_inds]);
            pl.plot(fc[rec_mr_inds])
            pl.subplot(222)
            pl.plot(ff[rec_mr_inds],fc[rec_mr_inds],'.')
            pl.show()

        if iter_dict['do_rec_flatm']:
            print('grad for rec flatm:')
            print('   ff mean: ', np.mean(ff[rec_flatm_inds]))
            print('   fc mean: ', np.mean(fc[rec_flatm_inds]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(ff[rec_flatm_inds],fc[rec_flatm_inds])
            print('   slope: ', slope, ' intercept: ', intercept, ' r_value: ', r_value)
            pl.subplot(221)
            pl.plot(ff[rec_flatm_inds]);
            pl.plot(fc[rec_flatm_inds])
            pl.subplot(222)
            pl.plot(ff[rec_flatm_inds],fc[rec_flatm_inds],'.')
            pl.show()
            
        return ff, fc
        
    #%% start of routine
    funcs = [L,gradL,S,gradS_list_line2,gradLnum,checkgrad,Huber,gradHuber,gradSnum,gradHubernum] # to allow access to the functions outside   
    
    Ndet = s.shape[0]
    Ntheta = s.shape[1]
    M = s.shape[2]
    Mflat = f.shape[1]
    Dx = x[1]-x[0]
    theta = iter_dict['theta']
    gamma = iter_dict['gamma']
        
    Radon_algo = iter_dict['Radon_algo']  
    gamma_abs = iter_dict['gamma_abs']
    gamma_phs = iter_dict['gamma_phs']
    lambda1 = iter_dict['lambda1']
    delta1 = iter_dict['delta1']
    lambda2 = iter_dict['lambda2']
    delta2 = iter_dict['delta2']
    weight = iter_dict['weights']
        
    if iter_dict['callback']:
        cbfun = callback
    else:
        cbfun = None

    #%% radon lookup table
    gradradon_inds = iter_dict['gradradon_inds']
    gradradon_vals = iter_dict['gradradon_vals']
    Radon_algo_args = (gradradon_inds,gradradon_vals)
    graddiffradon_inds = iter_dict['graddiffradon_inds']
    graddiffradon_vals = iter_dict['graddiffradon_vals']
    Radon_algo_args2 = (graddiffradon_inds,graddiffradon_vals)
    
    #%% mask for values outside the reconstruction circle
    r = Ndet/2-1.1-0.5
    xx, yy = np.meshgrid(np.linspace(-Ndet/2,Ndet/2,Ndet),np.linspace(-Ndet/2,Ndet/2,Ndet))
    mask = np.sqrt((xx+0.0)**2+(yy-0.0)**2)<=r
    mask = np.array(1)
    mask = mask.flatten()
    
    #%% normalisation factor for the cost function
    C = np.sum(s**2)/1

    #%% indices    
    if iter_dict['do_rec_slice_abs']:
        rec_slice_abs_inds = np.arange(Ndet*Ndet) # this is abs slice for separate rec & combined for single shot
    if iter_dict['do_rec_slice_phs']:
        rec_slice_phs_inds = iter_dict['do_rec_slice_abs']*Ndet*Ndet+np.arange(Ndet*Ndet)
    if iter_dict['do_rec_mo']:
        rec_mo_inds = iter_dict['do_rec_slice_abs']*Ndet*Ndet+iter_dict['do_rec_slice_phs']*Ndet*Ndet+np.arange(M*Ntheta)
    if iter_dict['do_rec_icfac']:
        rec_icfac_inds = iter_dict['do_rec_slice_abs']*Ndet*Ndet+iter_dict['do_rec_slice_phs']*Ndet*Ndet+iter_dict['do_rec_mo']*M*Ntheta
    if iter_dict['do_rec_gamma']:
        rec_gamma_inds = iter_dict['do_rec_slice_abs']*Ndet*Ndet+iter_dict['do_rec_slice_phs']*Ndet*Ndet+iter_dict['do_rec_mo']*M*Ntheta+iter_dict['do_rec_icfac']
    if iter_dict['do_rec_mr']:
        rec_mr_inds = iter_dict['do_rec_slice_abs']*Ndet*Ndet+iter_dict['do_rec_slice_phs']*Ndet*Ndet+iter_dict['do_rec_mo']*M*Ntheta+iter_dict['do_rec_icfac']+iter_dict['do_rec_gamma']+np.arange(Ndet-0)
    if iter_dict['do_rec_flatm']:
        rec_flatm_inds = iter_dict['do_rec_slice_abs']*Ndet*Ndet+iter_dict['do_rec_slice_phs']*Ndet*Ndet+iter_dict['do_rec_mo']*M*Ntheta+iter_dict['do_rec_icfac']+iter_dict['do_rec_gamma']+iter_dict['do_rec_mr']*(Ndet-0)+np.arange(Mflat)
        
    
    Nrecvalues = int(iter_dict['do_rec_slice_abs']*Ndet*Ndet+iter_dict['do_rec_slice_phs']*Ndet*Ndet\
                     +iter_dict['do_rec_mo']*M*Ntheta+iter_dict['do_rec_icfac']+iter_dict['do_rec_gamma']\
                     +iter_dict['do_rec_mr']*(Ndet-0))+iter_dict['do_rec_flatm']*Mflat
    

    #%% starting & fixed values
    start = np.zeros(Nrecvalues)
    
    # abs slice
    if np.any(iter_dict['slice_abs']): # input provided
        if iter_dict['do_rec_slice_abs']: # do rec 
            start[rec_slice_abs_inds] = iter_dict['slice_abs']
        else:
            slice_abs = iter_dict['slice_abs']
    else: # no input provided
        if iter_dict['do_rec_slice_abs']: # do rec  
            start[rec_slice_abs_inds] = np.zeros(len(rec_slice_abs_inds))
        else: 
            slice_abs = np.zeros((Ndet*Ndet))
    
    
    # phs slice
    if np.any(iter_dict['slice_phs']): # input provided
        if iter_dict['do_rec_slice_phs']: # do rec 
            start[rec_slice_phs_inds] = iter_dict['slice_phs']
        else:
            slice_phs = iter_dict['slice_phs']
    else: # no input provided
        if iter_dict['do_rec_slice_phs']: # do rec  
            start[rec_slice_phs_inds] = np.zeros(len(rec_slice_phs_inds))
        else: 
            slice_phs = np.zeros((Ndet*Ndet))
    
    # mo
    if np.any(iter_dict['mo']): # input provided
        if iter_dict['do_rec_mo']: # do rec of mo
            start[rec_mo_inds] = iter_dict['mo']
        else:
            mo = iter_dict['mo']
    else: # no input provided
        if iter_dict['do_rec_mo']: # do rec of mo
            start[rec_mo_inds] = np.zeros(len(rec_mo_inds))
        else: 
            mo = np.zeros(1)
            
    # gamma
    if iter_dict['do_rec_slice_phs']:
        gamma = 1.
    else:
        if iter_dict['gamma']: # input provided
            if iter_dict['do_rec_gamma']: # do rec of gamma
                start[rec_gamma_inds] = iter_dict['gamma']
            else:
                gamma = iter_dict['gamma']
        else: # no input provided
            if iter_dict['do_rec_gamma']: # do rec of gamma
                start[rec_gamma_inds] = 1.
            else: 
                gamma = 1.
                
    # icfac
    if np.any(iter_dict['icfac']): # input provided
        if iter_dict['do_rec_icfac']: # do rec of icfac
            start[rec_icfac_inds] = iter_dict['icfac']
        else:
            icfac = iter_dict['icfac']
    else: # no input provided
        if iter_dict['do_rec_icfac']: # do rec of icfac
            start[rec_icfac_inds] = np.zeros(len(rec_icfac_inds))
        else: 
            icfac = 1.
                
    # mr
    if np.any(iter_dict['mr']): # input provided 
        if iter_dict['do_rec_mr']: # do rec of mr
            start[rec_mr_inds] = iter_dict['mr']
        else:
            mr = iter_dict['mr']
    else: # no input provided
        if iter_dict['do_rec_mr']: # do rec of mo
            start[rec_mr_inds] = np.zeros(len(rec_mr_inds))
        else: 
            mr = np.zeros(Ndet)
            
    # flatm
    if np.any(iter_dict['flatm']): # input provided
        if iter_dict['do_rec_flatm']: # do rec of mr
            start[rec_mr_inds] = iter_dict['flatm']
        else:
            flatm = iter_dict['flatm']
    else: # no input provided
        if iter_dict['do_rec_flatm']: # do rec of flatm
            start[rec_flatm_inds] = np.zeros(len(rec_flatm_inds))
        else: 
            flatm = np.zeros(Mflat)    
            
    #%% bounds
    lbounds, ubounds =  np.expand_dims([],1), np.expand_dims([],1)
    if iter_dict['do_rec_slice_abs']:
        lbounds = iter_dict['slice_abs_lbound']*np.ones((Ndet*Ndet,1))
        ubounds = iter_dict['slice_abs_ubound']*np.ones((Ndet*Ndet,1))
    if iter_dict['do_rec_slice_phs']:
        lbounds = np.vstack((lbounds,iter_dict['slice_phs_lbound']*np.ones((Ndet*Ndet,1))))
        ubounds = np.vstack((ubounds,iter_dict['slice_phs_ubound']*np.ones((Ndet*Ndet,1))))
    if iter_dict['do_rec_mo']:
        lbounds = np.vstack((lbounds,iter_dict['mo_lbound']*np.ones((Ntheta,1))))
        ubounds = np.vstack((ubounds,iter_dict['mo_ubound']*np.ones((Ntheta,1))))
    if iter_dict['do_rec_icfac']:
        lbounds = np.vstack((lbounds,iter_dict['icfac_lbound']*np.ones((1,1))))
        ubounds = np.vstack((ubounds,iter_dict['icfac_ubound']*np.ones((1,1))))
    if iter_dict['do_rec_gamma']:
        lbounds = np.vstack((lbounds,iter_dict['gamma_lbound']*np.ones((1,1))))
        ubounds = np.vstack((ubounds,iter_dict['gamma_ubound']*np.ones((1,1))))
    if iter_dict['do_rec_mr']:
        lbounds = np.vstack((lbounds,iter_dict['fmo_lbound']*np.ones((Ndet,1))))
        ubounds = np.vstack((ubounds,iter_dict['fmo_ubound']*np.ones((Ndet,1))))        
    if iter_dict['do_rec_flatm']:
        lbounds = np.vstack((lbounds,iter_dict['flatm_lbound']*np.ones((Mflat,1))))
        ubounds = np.vstack((ubounds,iter_dict['flatm_ubound']*np.ones((Mflat,1))))        
    bounds = np.hstack((lbounds,ubounds))
    
    #%% minimisation
    print('gamma used for iteration: ', gamma)
    
    tt = time.time()
    res3 = optimize.fmin_l_bfgs_b(L, start, bounds=bounds, fprime=gradL,
                                  m=iter_dict['m'], factr=iter_dict['factr'], pgtol=1e-10, epsilon=1e-08, iprint=-1,
                                  maxfun=15000, maxiter=iter_dict['maxiter'], disp=0, callback=cbfun,
                                  maxls=iter_dict['maxls'])
    
    print(res3[2]['task'])
    print('iterations done: ', res3[2]['nit'])
    print("time for iteration: ",time.time()-tt)
    
    print("")
    print("L(p)      : {0:.3e}".format(L(res3[0])))
    print("S(p)      : {0:.3e}".format(S(res3[0])))
    if iter_dict['do_rec_slice_abs']:
        print("l1*H1(abs): {0:.3e}".format(lambda1*Huber(res3[0][rec_slice_abs_inds],delta1)))
    if iter_dict['do_rec_slice_phs']:
        print("l2*H2(phs): {0:.3e}".format(lambda2*Huber(res3[0][rec_slice_phs_inds],delta2)))
    print("")
    
    rec_dict = gen_rec_dict()
    rec_dict['Nrecvalues'] = Nrecvalues
    rec_dict['BFGS_results'] = res3
    if iter_dict['do_rec_slice_abs']:
        rec_dict['rec_slice_abs'] = np.reshape(res3[0][rec_slice_abs_inds],((Ndet,Ndet)))
        rec_dict['rec_slice_abs_inds'] = rec_slice_abs_inds
        print('mean rec abs slice: {0:.3e}'.format(np.mean(np.abs(res3[0][rec_slice_abs_inds]))))
    if iter_dict['do_rec_slice_phs']:
        rec_dict['rec_slice_phs'] = np.reshape(res3[0][rec_slice_phs_inds],((Ndet,Ndet)))
        rec_dict['rec_slice_phs_inds'] = rec_slice_phs_inds
        print('mean rec phs slice: ', np.mean(np.abs(res3[0][rec_slice_phs_inds])))
    if iter_dict['do_rec_mo']:
        rec_dict['rec_mo'] = res3[0][rec_mo_inds]
        rec_dict['rec_mo_inds'] = rec_mo_inds
        print('mean  mo: ', np.mean(res3[0][rec_mo_inds]))    
    if iter_dict['do_rec_gamma']:
        rec_dict['rec_gamma'] = res3[0][rec_gamma_inds]    
        rec_dict['rec_gamma_inds'] = rec_gamma_inds
        print('gamma: ', res3[0][rec_gamma_inds])
    if iter_dict['do_rec_icfac']:    
        rec_dict['rec_icfac'] = res3[0][rec_icfac_inds]
        rec_dict['rec_icfac_inds'] = rec_icfac_inds
        print('icfac: ', res3[0][rec_icfac_inds])
    if iter_dict['do_rec_mr']:
        rec_dict['rec_mr'] = res3[0][rec_mr_inds]
        rec_dict['rec_mr_inds'] = rec_mr_inds
        print('mean mr: ', np.mean(res3[0][rec_mr_inds]))
    if iter_dict['do_rec_flatm']:
        rec_dict['rec_flatm'] = res3[0][rec_flatm_inds]
        rec_dict['rec_flatm_inds'] = rec_flatm_inds
        print('mean flatm: ', np.mean(res3[0][rec_flatm_inds]))

    if iter_dict['callback']:
        rec_dict['Liter'] = Liter

    return rec_dict, funcs
    
def shiftimg(image,xshift=0,yshift=0):
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    f2 = interpolate.interp2d(x, y, image, kind='linear', copy=True, bounds_error=False, fill_value=0)
    xnew = x+xshift
    ynew = y+yshift
    return f2(xnew,ynew)


def n2s(number,length=3):
    a = str(number)
    if (len(a)<length):
        b = '0'
        for isi in range(length-len(a)-1):
            b = b + '0'    
        a =  b + a
    return a