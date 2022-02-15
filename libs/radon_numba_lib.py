# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:00:03 2022

@author: Peter Modregger

version 5.5 forked from iterlib

Containts the numba versions of radon_linear & radon_linear_grad

"""

import numpy as np
import numpy.matlib
from numba import jit


@jit(nopython=True,fastmath=True)
def radon_linear(data,theta,RA): # radon transform
    """
    USAGE:
        out = radon_numba_linear(data,theta,RA)

        data   - values: (0,np.inf),  shape: (Ndet,Ndet),   type: float64 (float32 possible)
        theta  - values: (0,2*np.pi), shape: (Ntheta),      type: float64 (float32 possible)
        RA     - values: (0,np.inf),  shape: (1),           type: float64 (float32 possible)
        
        out    - values: (0,np.inf),  shape: (Ndet,Ntheta), type: float64 (float32 possible)
        
        version 5.4
    """ 

    # print("RNL version 5.4")
    Ndet = data.shape[0]
    Ntheta = len(theta)
    n = 1.5 # this was found empirically, matching astra radon the best
    
    out = np.zeros((Ndet,Ntheta))
    cost = np.cos(theta)
    sint = np.sin(theta)
    
    for isi in range(Ndet*Ndet):
        
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
  
        a = np.all( (t>0) * (t<(Ndet-2)) ) # all t's must be between 0 & Ndet-2
        if a:
            d = data[isy,isx] # minimise mem access during next loop
            for ist in range(Ntheta):
                tt = np.int32(t[ist])
           
                
                d1n = (t[ist]-tt)**n # this here is an interpolation
                d2n = (tt+1-t[ist])**n
                
                b = 1/(d1n+d2n)
                out[tt,ist] += d*b*d2n
                out[tt+1,ist] += d*b*d1n
    return out 

# this is the gradient for combined contrast radon
@jit(nopython=True,fastmath=True)
def radon_linear_grad(data1,data2,theta,RA):
    """

    version 5.4
    """
    
    Ndet = data1.shape[0]
    Ntheta = len(theta)
    n = 1.5 # this was found empirically, matching astra radon the best
    
    out = np.zeros((Ndet,Ndet))
    cost = np.cos(theta)
    sint = np.sin(theta)
    
    for isi in range(Ndet*Ndet):
        
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