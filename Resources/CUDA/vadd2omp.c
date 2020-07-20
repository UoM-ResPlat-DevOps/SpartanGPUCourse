#include <stdio.h>
#include <omp.h>

void vadd2(int n, float * a, float * b, float * c)
{
    #pragma omp target map(to:n,a[0:n],b[0:n]) map(from:c[0:n])
    #pragma omp teams distribute parallel for simd
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}
