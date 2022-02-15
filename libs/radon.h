#ifndef radon_h__
#define radon_h__
 
extern void radon(double *data, double *out, double *theta, int Ndet, int Ntheta, double RA);

extern void gradradon(double *data1, double *data2, double *out, double *theta, int Ndet, int Ntheta, double RA);
 
#endif  // radon_h__
