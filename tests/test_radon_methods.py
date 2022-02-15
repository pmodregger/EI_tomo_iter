import numpy as np
import pylab as pl
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
import time
import radon_numba_lib as rn
import radon_omp_lib as rno


#%% setting up the data and calling the function
Ndet = 90
RA = Ndet//2+.4
theta = np.deg2rad(np.linspace(0,360,Ndet//2,endpoint=False))

#data_org = np.load('shepp_logan.npy')
#data = np.flipud(rescale(data_org, scale=Ndet/400, mode='reflect',multichannel=False))

data = shepp_logan_phantom()
data = rescale(data, scale=Ndet/400, mode='reflect')

t1_numba = time.perf_counter()
sino_numba = rn.radon_linear(data,theta,RA)
grad_numba = rn.radon_linear_grad(sino_numba,sino_numba,theta,RA)
t2_numba = time.perf_counter()

# repeat both steps with omp version
t1_omp = time.perf_counter()
sino_omp = rno.radon_linear(data,theta,RA)
grad_omp = rno.radon_linear_grad(sino_omp,sino_omp,theta,RA)
t2_omp = time.perf_counter()


print(Ndet, "-->", t2_numba-t1_numba, " (numba)")
print(Ndet, "-->", t2_omp-t1_omp, "(omp)")
print("relative deviation in sino: ", np.linalg.norm(sino_omp-sino_numba)/np.linalg.norm(sino_numba))
print("relative deviation in grad: ", np.linalg.norm(grad_omp-grad_numba)/np.linalg.norm(grad_numba))


pl.subplot(121)
pl.imshow(grad_numba)
pl.colorbar()
pl.subplot(122)
pl.imshow(grad_omp)
pl.colorbar()
pl.show()
