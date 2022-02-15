# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:26:03 2017

@author: Peter Modregger

version 5.3 moved calc_t out of the radon numba's for easier maintance
version 5.2 cos(atan(phi)) in calc of t implemented 
version 5.1 iteration now calls radon transform 2 instead of 4 times each step
version 5.0 numba jit implementation of radon transform

"""

from __future__ import print_function, division
import numpy as np
import pylab as pl
from scipy import optimize
import time
from scipy import interpolate
from skimage.transform import iradon
from numba import jit

def iter_parameters():
    """
    USAGE:
        iter_dict = iterlib.iter_parameters()
        
    DESCRIPTION:
        returns a default dictionary of parameters for the iteration
    
    """

    iter_dict = {}
    
    # parameter for the l-bfgs-b minimisation
    iter_dict['maxiter'] = 1000 # max number of iteration steps
    iter_dict['m'] = 100 # number of matrix corrections for BFGS; try m=100 - DONT USE m=1000 with fastmath... kills the kernel but no clue why
    iter_dict['maxls'] = 3 # number of line search steps for BFGS; try maxls=2,3
    iter_dict['factr'] = 1e9 # 1e9 works fine and fast, smaller = better results
    iter_dict['callback'] = 0 # show updates after each iteration step
    
    # switches for different retrievable values
    iter_dict['do_rec_slice_abs'] = 1 # retrieve absorption slice?
    iter_dict['do_rec_slice_phs'] = 0 # 0 -> single shot, 1 -> two shots
    iter_dict['do_rec_mo'] = 0 # retrieve mask offset position?
    iter_dict['do_rec_mr'] = 0 # retrieve ring artefact suppression position?
    
    # start or fixed values for retrievable values (works fine without giving starting values)
    iter_dict['slice_abs'] = [] # leave empty to not provide a starting/fixed value
    iter_dict['slice_phs'] = [] # start will be constructed out of these if appropriate
    iter_dict['mo'] = [] # offset on IC per projection variable
    iter_dict['mr'] = [] # ring removal variable
    
    # lower & upper bounds for retrievable values
    iter_dict['slice_abs_lbound'] = 0
    iter_dict['slice_abs_ubound'] = np.inf
    iter_dict['slice_phs_lbound'] = 0
    iter_dict['slice_phs_ubound'] = np.inf
    iter_dict['mo_lbound'] = -1
    iter_dict['mo_ubound'] = 1
    iter_dict['mr_lbound'] = -1 
    iter_dict['mr_ubound'] = 1
    
    # parameters for foward model
    iter_dict['gamma_abs'] = 1 # use absorption data?
    iter_dict['gamma_phs'] = 1 # use phase contrast data?
    
    # noise suppression parameters
    iter_dict['lambda1'] = 0 # for l2 norm should be at least 10*mean of rec
    iter_dict['delta1'] = 100
    iter_dict['lambda2'] = 0 # 2nd lambda for phase slice if reconstructed independently
    iter_dict['delta2'] = 100
    iter_dict['lambdatv'] = 0 # lambda for total energy variation
    
    
    # parameters of tomographic reconstruction
    iter_dict['RA'] = 128 # this is needed if gradradon_x are not given
    iter_dict['theta'] = 0 # rotation angles in radians
    iter_dict['weights'] = 1 #  leave at 1
    
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
    rec_dict['rec_mr'] = []
    rec_dict['rec_mr_inds'] = []
    rec_dict['Liter'] = []
    
    return rec_dict

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

@jit(nopython=True,fastmath=True)
def calc_t(isi,Ndet,RA,cost,sint): # calc indeces on sinogram
    
    isy, isx = isi//Ndet, isi%Ndet
    xs = isx-Ndet/2+0.5
    ys = isy-RA
    r = np.sqrt(xs**2+ys**2)
    
    # this is too avoid a div by 0 error
    if xs == 0:
        if ys == 0:
            sqrtu = 1
            usqrtu = 0
        elif ys > 0:
            sqrtu = 0
            usqrtu = 1
        else:
            sqrtu = 0
            usqrtu = -1
    else:
        u = ys/xs
        sqrtu = np.sign(xs)/np.sqrt(1+u**2)
        usqrtu = u*sqrtu
    
    t = r*(sqrtu*cost-usqrtu*sint)+RA
    
    return t, isx, isy
    
# this is a radon transform
@jit(nopython=True,fastmath=True)
def radon_numba_linear(data,theta,RA): # radon transform
    """
    USAGE:
        out = radon_numba_linear(data,theta,RA)

        data   - values: (0,np.inf),  shape: (Ndet,Ndet),   type: float64 (float32 possible)
        theta  - values: (0,2*np.pi), shape: (Ntheta),      type: float64 (float32 possible)
        RA     - values: (0,np.inf),  shape: (1),           type: float64 (float32 possible)
        
        out    - values: (0,np.inf),  shape: (Ndet,Ntheta), type: float64 (float32 possible)
        
        version 5.2
    """ 

    Ndet = data.shape[0]
    Ntheta = len(theta)
    n = 1.5 # this was found empirically, matching astra radon the best
    
    out = np.zeros((Ndet,Ntheta))
    cost = np.cos(theta)
    sint = np.sin(theta)
    
    for isi in range(Ndet*Ndet):
        
        t, isx, isy = calc_t(isi,Ndet,RA,cost,sint)
  
        a = np.all( (t>0) * (t<(Ndet-2)) ) # all t's must be between 0 & Ndet-2
        if a:
            for ist in range(Ntheta):
                tt = np.int32(t[ist])
           
                d1n = (t[ist]-tt)**n  # this here is an interpolation
                d2n = (tt+1-t[ist])**n
                
                b = 1/(d1n+d2n)
                out[tt,ist] += data[isy,isx]*b*d2n
                out[tt+1,ist] += data[isy,isx]*b*d1n
    return out

# this is the gradient for standard tomo
@jit(nopython=True,fastmath=True)
def radon_numba_linear_grad_0(data,theta,RA):
    """

    version 5.2
    """
    
    Ndet = data.shape[0]
    Ntheta = len(theta)
    n = 1.5 # this was found empirically, matching astra radon the best
    
    out = np.zeros((Ndet,Ndet))
    cost = np.cos(theta)
    sint = np.sin(theta)
    
    for isi in range(Ndet*Ndet):
        
        t, isx, isy = calc_t(isi,Ndet,RA,cost,sint)
        
        a = np.all( (t>0) * (t<(Ndet-2)) )
        if a:
            for ist in range(Ntheta):
                tt = np.int32(t[ist])
           
                d1n = (t[ist]-tt)**n
                d2n = (tt+1-t[ist])**n
                
                # grad + grad diff radon transform
                out[isy,isx] += (data[tt,ist]*d2n + data[tt+1,ist]*d1n)/(d1n+d2n)

    return out

# this is the gradient for single shot tomo
@jit(nopython=True,fastmath=True)
def radon_numba_linear_grad(data1,data2,theta,RA):
    """

    version 5.2
    """
    
    Ndet = data1.shape[0]
    Ntheta = len(theta)
    n = 1.5 # this was found empirically, matching astra radon the best
    
    out = np.zeros((Ndet,Ndet))
    cost = np.cos(theta)
    sint = np.sin(theta)
    
    for isi in range(Ndet*Ndet):
        
        t, isx, isy = calc_t(isi,Ndet,RA,cost,sint)
        
        a = np.all( (t>0) * (t<(Ndet-2)) )
        if a:
            for ist in range(Ntheta):
                tt = np.int32(t[ist])
           
                d1n = (t[ist]-tt)**n
                d2n = (tt+1-t[ist])**n
                
                # grad + grad diff radon transform
                out[isy,isx] += ( (data1[tt,ist]+data2[tt,ist])*d2n + 
                                 data2[tt+1,ist]*(d1n-d2n)+(data1[tt+1,ist]-data2[tt+2,ist])*d1n)/(d1n+d2n)

    return out


def TV(image): # total energy variation
    return np.sum(np.diff(image,axis=0)**2) + np.sum(np.diff(image,axis=1)**2)

def gradTV(image): # gradient of total energy variation
    tempy = np.zeros(image.shape)
    tempy[:-1,:] = np.diff(image,axis=0)
    tempx = np.zeros(image.shape)
    tempx[:,:-1] = np.diff(image,axis=1)
    return 2*(np.roll(tempx,1,axis=1)-tempx+np.roll(tempy,1,axis=0)-tempy)

def gradTVnum(image): # numerical gradient of total energy variation for testing
    gradTVnum_ = np.zeros(image.size)
    TV_ = TV(image)
    p = image.flatten()
    fac = 1e-6
    
    for isn in range(image.size):                
        p0 = p.copy();
        p0[isn] = p[isn]+fac
        gradTVnum_[isn] = (TV(np.reshape(p0,(image.shape)))-TV_)/fac
        # p0[isn] = p[isn]
        
    return gradTVnum_    

# iter_tomo_rec is a test function for standard tomographic reconstruction (eg. absorption tomography).
def iter_tomo_rec(expsino,iter_dict):
    """
    
    USAGE:    
        rec_dict, funcs = iter_tomo_rec(expsino,iter_dict)
        
    TO ACCESS THE Reconstruction
        rec = rec_dict['rec_slice_abs']
    
    """
    # COST FUNCTION
    # @jit(nopython=True,fastmath=True) DONT USE MAKES EVERYTHING SLOWER
    def L(p):
        global pradon
        """
        theta in rad
        """
        pradon = radon_numba_linear(np.reshape(p,((Ndet,Ndet))),theta,RA)
        return np.sum((expsino-pradon)**2)*C
        
    # GRADIENT OF COST FUNCTION
    # @jit(nopython=True,fastmath=True) DONT USE MAKES EVERYTHING SLOWER
    def gradL(p):
        global pradon
        temp = -2*((expsino-pradon))*C
        return radon_numba_linear_grad_0(temp,theta,RA).flatten()
    
    # NUMERICAL GRADIENT OF COST FUNCTION FOR TESTING
    # @jit(nopython=True,fastmath=True)
    def gradLnum(p):
    
        gradLnum_ = np.zeros(len(p))
        L_ = L(p)
        for isn in range(len(p)):
            p0 = p.copy(); p0[isn] = p[isn]+1e-7
            gradLnum_[isn] = (L(p0)-L_)/1e-7
            
        return gradLnum_

    def callback(p):
        pl.imshow(np.reshape(p,((Ndet,Ndet))))
        pl.show()
    
    #%% Routine starts here 
    
    # Gathering data from call
    Ndet = expsino.shape[0]
    theta = iter_dict['theta']
    RA = iter_dict['RA']    
    Nrecvalues = Ndet*Ndet
    start = iter_dict['slice_abs']
    if not np.any(start): # start value was given as 0    
        start = np.zeros(Nrecvalues).flatten()    
    else:
        start = start
    
    C = 1/np.sum(expsino**2) # normalisation factor for cost function
    
    if iter_dict['callback']:
        cbfun = callback
    else:
        cbfun = None
    
    # bounds
    lbounds = np.zeros((Ndet*Ndet,1))
    ubounds = np.inf*np.ones((Ndet*Ndet,1))
    bounds = np.hstack((lbounds,ubounds))
    
    #%% minimisation
    tt = time.time()
    print('about to start iterion')
    res3 = optimize.fmin_l_bfgs_b(L, start, bounds=bounds, fprime=gradL,
                                  m=iter_dict['m'], factr=iter_dict['factr'], pgtol=1e-10, epsilon=1e-08, iprint=0,
                                  maxfun=15000, maxiter=iter_dict['maxiter'], disp=0, callback=cbfun,
                                  maxls=iter_dict['maxls'])
    
    print(res3[2]['task'])
    print('iterations done: ', res3[2]['nit'])
    print("Cost function value L(rec): ", L(res3[0]))
    print("time estimate: ",time.time()-tt)
    
    
    funcs = [L,gradL,gradLnum] # to allow access to the functions outside   
    
    rec_dict = gen_rec_dict()
    rec_dict['Nrecvalues'] = Nrecvalues
    rec_dict['BFGS_results'] = res3
    if iter_dict['do_rec_slice_abs']:
        rec_dict['rec_slice_abs'] = np.reshape(res3[0],((Ndet,Ndet)))
    
    return rec_dict, funcs


#@profile
def iter_single_shot(x,f,m,s,iter_dict):
    """
    USAGE:
         rec_dict, funcs = iterlib.iter_single_shot(mflat,ICflat,m0,ICsamp,iter_dict)
         
    It says single shot, but separated reconstruction of absorption and phase is implemented
    """
    
    global Niter, Liter
    
    Niter = 0
    Liter = []
    
    def L(p): # total cost function
    
        out = S(p)
        if iter_dict['lambda1'] > 0:
            out += lambda1*Huber(p[rec_slice_abs_inds],delta1)
        if iter_dict['lambda2'] > 0:
            out += lambda2*Huber(p[rec_slice_phs_inds],delta2)
        if iter_dict['lambdatv'] > 0:
            out += lambdatv*TV(np.reshape(p[rec_slice_abs_inds],(Ndet,Ndet)))
            
        return out
            
    def gradL(p): # gradient of total cost function
    
        out = gradS_list_line2(p)
        if iter_dict['lambda1'] > 0:
            out[rec_slice_abs_inds] += lambda1*gradHuber(p[rec_slice_abs_inds],delta1)
        if iter_dict['lambda2'] > 0:
            out[rec_slice_phs_inds] += lambda2*gradHuber(p[rec_slice_phs_inds],delta2)
        if iter_dict['lambdatv'] > 0:
            out[rec_slice_abs_inds] += lambdatv*gradTV(np.reshape(p[rec_slice_abs_inds],(Ndet,Ndet)))
            
        return out
    
    def S(p): # cost function of model part (slice(s), mr, mo)
        global fradon, Dradon # that is the radon transform of the current slice
        
        if iter_dict['do_rec_mo']:
            mo = np.reshape(p[rec_mo_inds],(Ntheta,M))
        else:
            mo = 0
        if iter_dict['do_rec_mr']:
            mr = p[rec_mr_inds]
        else:
            mr = np.zeros(Ndet)
        
        if iter_dict['do_rec_slice_abs']:
            fradon = radon_numba_linear(np.reshape(p[rec_slice_abs_inds],((Ndet,Ndet))),theta,RA)
            Aradon = np.exp(-gamma_abs*fradon)
        else:
            Aradon = np.ones((Ndet,Ntheta))        
        if iter_dict['do_rec_slice_phs']:
            Dradon = np.zeros((Ndet,Ntheta))
            Dradon[1:,:] = gamma_phs*np.diff(radon_numba_linear(np.reshape(slice_phs,((Ndet,Ndet))),theta,RA,0),1,0)
        else: # single shot
            Dradon = np.zeros((Ndet,Ntheta))    
            Dradon[1:,:] =  gamma*gamma_phs*np.diff(fradon,1,0)
            
        fs = np.zeros((Ndet,Ntheta,M))
        for isp in range(Ndet):
            fs[isp,:,:] = np.matlib.repmat(Aradon[isp,:],M,1).transpose()*\
                        np.interp(m-mo-mr[isp]+np.matlib.repmat(Dradon[isp,:],M,1).transpose(),x,f[isp,:])
                        
        # for isp in range(Ndet):
        #     for ist in range(Ntheta):
        #         fs[isp,ist,:] = np.matlib.repmat(Aradon[isp,ist],M,1).transpose()*\
        #                     np.interp(m-mo[ist]-mr[isp]+np.matlib.repmat(Dradon[isp,ist],M,1).transpose(),x,f[isp,:])
                                        
        # jitter_corrections[istheta]*Ndither
        return np.sum(weight*(s-fs) **2)/C

    def gradS_list_line2(p): # gradient of cost function of radon part 
        global fradon, Dradon
        
        # if iter_dict['do_rec_slice_abs']:
        #     slice_abs = p[rec_slice_abs_inds]
        # if iter_dict['do_rec_slice_phs']:
            # slice_phs = p[rec_slice_phs_inds]
        if iter_dict['do_rec_mo']:
            mo = np.reshape(p[rec_mo_inds],(Ntheta,M))
        else:
            mo = 0
        if iter_dict['do_rec_mr']:
            mr = p[rec_mr_inds]
        else:
            mr = np.zeros(Ndet)
                   
        if iter_dict['do_rec_slice_abs']:
            # fradon = radon_numba_linear(np.reshape(slice_abs,((Ndet,Ndet))),theta,RA,0)
            Aradon = np.exp(-gamma_abs*fradon)
        else:
            Aradon = np.ones((Ndet,Ntheta))
        if not iter_dict['do_rec_slice_phs']:
            # Dradon = np.zeros((Ndet,Ntheta))
            # Dradon[1:,:] = gamma_phs*np.diff(radon_numba_linear(np.reshape(slice_phs,((Ndet,Ndet))),theta,RA,0),1,0)
        
            Dradon = np.zeros((Ndet,Ntheta))    
            Dradon[1:,:] =  gamma*gamma_phs*np.diff(fradon,1,0)
            
        fs = np.zeros((Ndet,Ntheta,M))
        fss = np.zeros((Ndet,Ntheta,M))
        ff1 = np.zeros((Ndet,Ntheta,M))
            
        for isp in range(Ndet):
            AAradon = np.matlib.repmat(Aradon[isp,:],M,1).transpose()
            DDradon = np.matlib.repmat(Dradon[isp,:],M,1).transpose()
            fs[isp,:,:]  = AAradon*np.interp(m-mo-mr[isp]+DDradon,x,f[isp,:])
            fss[isp,:,:] = AAradon*np.interp(m+Dx/2-mo-mr[isp]+DDradon,x[1:],np.diff(f[isp,:])/Dx)
            ff1[isp,:,:]  = AAradon*np.interp(m-mo-mr[isp]+DDradon,x,f[isp,:])    
        
        gradS_list_line = np.zeros(Nrecvalues)
        if iter_dict['do_rec_slice_abs']:
             
            temp1 =  np.reshape(2*np.sum((weight*(s-fs))*fs,2).flatten()*gamma_abs/C,((Ndet,Ntheta)))
            temp2 =  np.reshape(2*np.sum((weight*(fs-s))*fss,2).flatten()*gamma*gamma_phs/C,((Ndet,Ntheta)))
            
            if iter_dict['do_rec_slice_phs']: # do seperate absorption & phase reconstructions
                gradS_list_line[rec_slice_abs_inds] = radon_numba_linear(temp1,theta,RA,1).flatten()
                gradS_list_line[rec_slice_phs_inds] = radon_numba_linear(temp2,theta,RA,2).flatten()
            else: # single shot
                # gradS_list_line[rec_slice_abs_inds] = radon_numba_linear(temp1,theta,RA,1).flatten() + \
                                    # radon_numba_linear(temp2,theta,RA,2).flatten()
                gradS_list_line[rec_slice_abs_inds] = radon_numba_linear_grad(temp1,temp2,theta,RA).flatten()

        elif iter_dict['do_rec_slice_phs']:
            temp2 =  2*np.sum((weight*(fs-s))*fss,2).flatten()*gamma*gamma_phs/C            
            gradS_list_line[rec_slice_phs_inds] = radon_numba_linear(temp2,theta,RA,2).flatten()
            
        if iter_dict['do_rec_mo']:
            gradS_list_line[rec_mo_inds] = 2/C*(np.sum(weight*(s-fs)*fss,0)).flatten()
           
        if iter_dict['do_rec_mr']:
            gradS_list_line[rec_mr_inds] = 2/C*np.sum(weight*(s-fs)*fss,(1,2)).flatten()
        
        return gradS_list_line
    
    def Huber(p,delta):
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
    
    # @jit(nopython=True,fastmath=True)
    def gradLnum(p):
        gradLnum_ = np.zeros(len(p))
        L_ = L(p)
        for isn in range(len(p)):
            p0 = p.copy();
            if p0[isn] == 0:
                h = 1e-8
            else:
                h = p0[isn]*1e-8
            p0[isn] = p[isn]+h
            gradLnum_[isn] = (L(p0)-L_)/h
            
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
            pl.subplot(1,3,(1,2))
            pl.imshow(perc(np.reshape(p[rec_slice_abs_inds],(Ndet,Ndet))));pl.colorbar();
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
        pl.show()
        Niter += 1
    
 
        
    #%% start of routine
    funcs = [L,gradL,S,gradS_list_line2,gradLnum,Huber,gradHuber,gradSnum,gradHubernum] # to allow access to the functions outside   
    
    Ndet = s.shape[0]
    Ntheta = s.shape[1]
    M = s.shape[2]
    Dx = x[1]-x[0]
    theta = iter_dict['theta']
    gamma = iter_dict['gamma']
    RA = iter_dict['RA']
         
    gamma_abs = iter_dict['gamma_abs']
    gamma_phs = iter_dict['gamma_phs']
    lambda1 = iter_dict['lambda1']
    delta1 = iter_dict['delta1']
    lambda2 = iter_dict['lambda2']
    delta2 = iter_dict['delta2']
    lambdatv = iter_dict['lambdatv']
    weight = iter_dict['weights']
        
    if iter_dict['callback']:
        cbfun = callback
    else:
        cbfun = None
    
    # normalisation factor for the cost function
    C = np.sum(s**2)

    #%% indices    
    if iter_dict['do_rec_slice_abs']:
        rec_slice_abs_inds = np.arange(Ndet*Ndet) # this is abs slice for separate rec & combined for single shot
    if iter_dict['do_rec_slice_phs']:
        rec_slice_phs_inds = iter_dict['do_rec_slice_abs']*Ndet*Ndet+np.arange(Ndet*Ndet)
    if iter_dict['do_rec_mo']:
        rec_mo_inds = iter_dict['do_rec_slice_abs']*Ndet*Ndet+iter_dict['do_rec_slice_phs']*Ndet*Ndet+np.arange(M*Ntheta)
    if iter_dict['do_rec_mr']:
        rec_mr_inds = iter_dict['do_rec_slice_abs']*Ndet*Ndet+iter_dict['do_rec_slice_phs']*Ndet*Ndet+iter_dict['do_rec_mo']*Ntheta+np.arange(Ndet)
        
    
    Nrecvalues = int(iter_dict['do_rec_slice_abs']*Ndet*Ndet+iter_dict['do_rec_slice_phs']*Ndet*Ndet\
                     +iter_dict['do_rec_mo']*M*Ntheta+iter_dict['do_rec_mr']*(Ndet))
    

    #%% setting starting or fixed values
    start = np.zeros(Nrecvalues)
    
    # absorption or combined slice
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
     
    # phase slice
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
    
    # mo - mask offset
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
                
    # mr - ring removal
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
               
    #%% bounds
    lbounds, ubounds =  np.expand_dims([],1), np.expand_dims([],1)
    if iter_dict['do_rec_slice_abs']:
        lbounds = iter_dict['slice_abs_lbound']*np.ones((Ndet*Ndet,1))
        ubounds = iter_dict['slice_abs_ubound']*np.ones((Ndet*Ndet,1))
    if iter_dict['do_rec_slice_phs']:
        lbounds = np.vstack((lbounds,iter_dict['slice_phs_lbound']*np.ones((Ndet*Ndet,1))))
        ubounds = np.vstack((ubounds,iter_dict['slice_phs_ubound']*np.ones((Ndet*Ndet,1))))
    if iter_dict['do_rec_mo']:
        lbounds = np.vstack((lbounds,iter_dict['mo_lbound']*np.ones((M*Ntheta,1))))
        ubounds = np.vstack((ubounds,iter_dict['mo_ubound']*np.ones((M*Ntheta,1))))
    if iter_dict['do_rec_mr']:
        lbounds = np.vstack((lbounds,iter_dict['mr_lbound']*np.ones((Ndet,1))))
        ubounds = np.vstack((ubounds,iter_dict['mr_ubound']*np.ones((Ndet,1))))         
    bounds = np.hstack((lbounds,ubounds))
    
    print('len(bounds): ',len(bounds))
    print('Nrecvalues: ',Nrecvalues)
    
    #%% minimisation
    print('iterlib version: 5.3')
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
    print("L(p)       : {0:.3e}".format(L(res3[0])))
    print("S(p)       : {0:.3e}".format(S(res3[0])))
    if iter_dict['do_rec_slice_abs']:
        print("l1*H1(abs) : {0:.3e}".format(lambda1*Huber(res3[0][rec_slice_abs_inds],delta1)))
        print("ltv*TV(abs): {0:.3e}".format(lambdatv*TV(np.reshape(res3[0][rec_slice_abs_inds],(Ndet,Ndet)))))
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
        print('mean mo: ', np.mean(res3[0][rec_mo_inds]))    
    if iter_dict['do_rec_mr']:
        rec_dict['rec_mr'] = res3[0][rec_mr_inds]
        rec_dict['rec_mr_inds'] = rec_mr_inds
        print('mean mr: ', np.mean(res3[0][rec_mr_inds]))

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

def perc(I, p = 99):
    "Usage: I = pm.perc(I, p = 95); percentile an image"
    zhih = np.percentile(I, p)
    zlow = np.percentile(I, 100-p)
    II = I.copy()
    II[I>zhih] = zhih
    II[I<zlow] = zlow
    return II
