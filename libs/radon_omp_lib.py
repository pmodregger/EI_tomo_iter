# -*- coding: utf-8 -*-
"""

@author: Tomasz Korzec

python wrappers for c based openmp

"""

import numpy as np
import numpy.ctypeslib as ctl
import ctypes
import pathlib

def radon_linear(data,theta,RA): # radon transform
    """
    USAGE:
        out = radon_numba_linear(data,theta,RA)

        data   - values: (0,np.inf),  shape: (Ndet,Ndet),   type: float64 (float32 possible)
        theta  - values: (0,2*np.pi), shape: (Ntheta),      type: float64 (float32 possible)
        RA     - values: (0,np.inf),  shape: (1),           type: float64 (float32 possible)
        
        out    - values: (0,np.inf),  shape: (Ndet,Ntheta), type: float64 (float32 possible)
    """
    
    Ndet = data.shape[0]
    Ntheta = len(theta)
    
    out = np.zeros((Ndet,Ntheta))
    
    libname = 'radon.so'
    libdir = pathlib.Path(__file__).parent.resolve()



    lib=ctl.load_library(libname, libdir)

    py_radon = lib.radon
    py_radon.argtypes = [ctl.ndpointer(np.float64, 
                                         flags='aligned, c_contiguous'), 
                         ctl.ndpointer(np.float64,      
                                         flags='aligned, c_contiguous'),
                         ctl.ndpointer(np.float64,
                                         flags='aligned, c_contiguous'),
                           ctypes.c_int, ctypes.c_int, ctypes.c_double]    

    py_radon(data, out, theta, Ndet, Ntheta, RA)
    return out


def radon_linear_grad(data1,data2,theta,RA):
    """
    USAGE:
        out = radon_omp_linear_grad(data1,data2,theta,RA)

        data1   - values: (0,np.inf),  shape: (Ndet,Ndet),   type: float64
        data2   - values: (0,np.inf),  shape: (Ndet,Ndet),   type: float64
        theta   - values: (0,2*np.pi), shape: (Ntheta),      type: float64
        RA      - values: (0,np.inf),  shape: (1),           type: float64
        
        out    - values: (0,np.inf),  shape: (Ndet,Ntheta), type: float64 (float32 possible)
    """
    Ndet = data1.shape[0]
    Ntheta = len(theta)
    
    out = np.zeros((Ndet,Ndet))
    
    libname = 'radon.so'
    libdir = pathlib.Path(__file__).parent.resolve()
    lib=ctl.load_library(libname, libdir)

    py_gradradon = lib.gradradon
    py_gradradon.argtypes = [ctl.ndpointer(np.float64, 
                                         flags='aligned, c_contiguous'),
                             ctl.ndpointer(np.float64, 
                                         flags='aligned, c_contiguous'),
                             ctl.ndpointer(np.float64,      
                                         flags='aligned, c_contiguous'),
                             ctl.ndpointer(np.float64,
                                         flags='aligned, c_contiguous'),
                             ctypes.c_int, ctypes.c_int, ctypes.c_double]    

    py_gradradon(data1, data2, out, theta, Ndet, Ntheta, RA)
    return out
