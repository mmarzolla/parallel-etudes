/****************************************************************************
 *
 * cuda-matsum.cu - Matrix-matrix addition
 *
 * Copyright (C) 2017--2024 Moreno Marzolla <https://www.unibo.it/sitoweb/moreno.marzolla/>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

/***
% HPC - Matrix-matrix addition
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla/)
% Last updated: 2024-01-04

The program [cuda-matsum.cu](cuda-matsum.cu) computes the sum of two
square matrices of size $N \times N$ using the CPU. Modify the program
to use the GPU; you must modify the function `matsum()` in such a way
that the new version is transparent to the caller, i.e., the caller is
not aware whether the computation happens on the CPU or the GPU. To
this aim, function `matsum()` should:

- allocate memory on the device to store copies of $p, q, r$;

- copy $p, q$ from the _host_ to the _device_;

- execute a kernel that computes the sum $p + q$;

- copy the result from the _device_ back to the _host_;

- free up device memory.

The program must work with any value of the matrix size $N$, even if
it nos an integer multiple of the CUDA block size. Note that there is
no need to use shared memory: why?

To compile:

        nvcc cuda-matsum.cu -o cuda-matsum -lm

To execute:

        ./cuda-matsum [N]

Example:

        ./cuda-matsum 1024

## Files

- [cuda-matsum.cu](cuda-matsum.cu)
- [hpc.h](hpc.h)

***/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "hpc.h"

#ifndef SERIAL
#define BLKDIM 32

__global__ void matsum_kernel( float *p, float *q, float *r, int n )
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<n && j<n )
        r[i*n + j] = p[i*n + j] + q[i*n + j];
}
#endif

void matsum( float *p, float *q, float *r, int n )
{
#ifdef SERIAL
    /* [TODO] Modify the body of this function to
       - allocate memory on the device
       - copy p and q to the device
       - call an appropriate kernel
       - copy the result from the device to the host
       - free memory on the device
    */
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            r[i*n + j] = p[i*n + j] + q[i*n + j];
        }
    }
#else
    float *d_p, *d_q, *d_r;	  /* device copies of p, q, r */
    const size_t size = n*n*sizeof(*p);
    dim3 block(BLKDIM, BLKDIM);
    dim3 grid((n+BLKDIM-1)/BLKDIM, (n+BLKDIM-1)/BLKDIM);

    /* Allocate space for device copies of p, q, r */
    cudaMalloc((void **)&d_p, size);
    cudaMalloc((void **)&d_q, size);
    cudaMalloc((void **)&d_r, size);

    /* Copy inputs to device */
    cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, size, cudaMemcpyHostToDevice);

    /* Launch matsum() kernel on GPU */
    matsum_kernel<<<grid, block>>>(d_p, d_q, d_r, n);

    /* Copy result back to host */
    cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);

    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_r);
#endif
}

/* Initialize square matrix p of size nxn */
void fill( float *p, int n )
{
    int k = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            p[i*n+j] = k;
            k = (k+1) % 1000;
        }
    }
}

/* Check result */
int check( float *r, int n )
{
    int k = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (fabsf(r[i*n+j] - 2.0*k) > 1e-5) {
                fprintf(stderr, "Check FAILED: r[%d][%d] = %f, expeted %f\n", i, j, r[i*n+j], 2.0*k);
                return 0;
            }
            k = (k+1) % 1000;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main( int argc, char *argv[] )
{
    float *p, *q, *r;
    int n = 1024;
    const int max_n = 5000;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_n ) {
        fprintf(stderr, "FATAL: the maximum allowed matrix size is %d\n", max_n);
        return EXIT_FAILURE;
    }

    printf("Matrix size: %d x %d\n", n, n);

    const size_t size = n*n*sizeof(*p);

    /* Allocate space for p, q, r */
    p = (float*)malloc(size); assert(p != NULL);
    fill(p, n);
    q = (float*)malloc(size); assert(q != NULL);
    fill(q, n);
    r = (float*)malloc(size); assert(r != NULL);

    const double tstart = hpc_gettime();
    matsum(p, q, r, n);
    const double elapsed = hpc_gettime() - tstart;

    printf("Elapsed time (including data movement): %f\n", elapsed);
    printf("Throughput (Melements/s): %f\n", n*n/(1e6 * elapsed));

    /* Check result */
    check(r, n);

    /* Cleanup */
    free(p);
    free(q);
    free(r);

    return EXIT_SUCCESS;
}
