#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include "radon.h"

#define OMP_THREADS 3

void radon(double *data, double *out, double *theta, int Ndet, int Ntheta, double RA)
{
   //double n = 1.5; // hard-coded for n=1.5
   double t[Ntheta];
   int isx, isy;
   double *d;
   double sintab[Ntheta],costab[Ntheta];
   double xs, ys, r, phi, cosphi, sinphi, z;
   char a;

   omp_set_num_threads(OMP_THREADS);
  
   //pre-tabulate sin/cos theta
   #pragma omp parallel for
   for(int it=0; it<Ntheta; it++)
   {
      sintab[it] = sin(theta[it]);
      costab[it] = cos(theta[it]);
   }

   d = data;
   for(isy=0; isy<Ndet; isy++)
   {
      ys = isy-RA;
      for(isx=0; isx<Ndet; isx++)
      {
         xs  = isx-Ndet*0.5+0.5;
         r   = sqrt(xs*xs + ys*ys);
         // should work for xs not 0:
         z = ys/xs;
         cosphi = (xs>0) ? 1.0/sqrt(1+z*z) : -1.0/sqrt(1+z*z);
         sinphi = z*cosphi;
         a = 1;

         #pragma omp parallel for reduction(*:a)
         for(int it=0; it<Ntheta; it++)
         {
            t[it] = RA + r*(costab[it]*cosphi - sintab[it]*sinphi);
            a *= (t[it]>0)&&(t[it]<Ndet-2);
         }

         if (a)
         {
            #pragma omp parallel
            {
               int tt;
               //double d1n, d2n, b;
               double d2d1,b;
               #pragma omp for
               for(int ist=0; ist<Ntheta; ist++)
               {
                  tt = (int) t[ist];
                  b = 1.0/(t[ist]-tt)-1.0;
                  d2d1 = sqrt(b*b*b);
                  out[tt*Ntheta+ist]     += *d/(1.0+1/d2d1);
                  out[(tt+1)*Ntheta+ist] += *d/(1.0+d2d1);
               }
            }
         }//if
         d++;
      }//isx
   }//isy
}

void gradradon(double *data1, double *data2, double *out, double *theta, int Ndet, int Ntheta, double RA)
{
   //double n = 1.5;   // hard-coded for n=1.5
   double t[Ntheta];
   int isx, isy;
   double sintab[Ntheta],costab[Ntheta];
   double xs, ys, r, phi, cosphi, sinphi, z;
   char a;
   double o;

   omp_set_num_threads(OMP_THREADS);
  
   //pre-tabulate sin/cos theta
   #pragma omp parallel for
   for(int it=0; it<Ntheta; it++)
   {
      sintab[it] = sin(theta[it]);
      costab[it] = cos(theta[it]);
   }

   for(isy=0; isy<Ndet; isy++)
   {
      ys = isy-RA;
      for(isx=0; isx<Ndet; isx++)
      {
         o = 0.0;
         xs  = isx-Ndet*0.5+0.5;
         r   = sqrt(xs*xs + ys*ys);
         // should work for xs not 0:
         z = ys/xs;
         cosphi = (xs>0) ? 1.0/sqrt(1+z*z) : -1.0/sqrt(1+z*z);
         sinphi = z*cosphi;
         a = 1;

         #pragma omp parallel for reduction(*:a)
         for(int it=0; it<Ntheta; it++)
         {
            t[it] = RA + r*(costab[it]*cosphi - sintab[it]*sinphi);
            a *= (t[it]>0)&&(t[it]<Ndet-2);
         }

         if (a)
         {
            #pragma omp parallel reduction (+:o)
            {
               int tt;
               double dt;
               double d1n, d2n;
               int idx;
               #pragma omp for
               for(int ist=0; ist<Ntheta; ist++)
               {
                  tt = (int) t[ist];
                  dt = t[ist] - (double) tt;
                  d1n = sqrt(dt*dt*dt);
                  dt = 1.0-dt;
                  d2n = sqrt(dt*dt*dt);
                               
                  idx = tt*Ntheta+ist;
                  o += ( (data1[idx]+data2[idx])*d2n + 
                         data2[idx+Ntheta]*(d1n-d2n) +
                         (data1[idx+Ntheta]-data2[idx+Ntheta+Ntheta])*d1n )/(d1n+d2n);
               }
            }
            out[isy*Ndet+isx] = o;
         }//if
      }//isx
   }//isy
}
