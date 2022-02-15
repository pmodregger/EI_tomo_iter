# compile into shared object
gcc -Ofast -mavx -mtune=native  -march=native -shared -o radon_omp_lib.so -fPIC -fopenmp radon_omp_lib.c -lm
 
# compile test for profiling
#gcc -c -O3 -fopenmp radon.c -lm
#gcc -c test.c -lm
#gcc -o test test.o radon.o -fopenmp -lm

