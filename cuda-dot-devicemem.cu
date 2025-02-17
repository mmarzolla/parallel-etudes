/****************************************************************************
 *
 * cuda-dot-devicemem.cu - Dot product with CUDA using __device__ memory
 *
 * Copyright (C) 2017--2021 Moreno Marzolla <https://www.unibo.it/sitoweb/moreno.marzolla/>
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
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-dot-devicemem.cu -o cuda-dot-devicemem -lm
 *
 * Run with:
 * ./cuda-dot-devicemem [len]
 *
 * Example:
 * ./cuda-dot-devicemem
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define BLKDIM 1024

__device__ double d_tmp[BLKDIM];

__global__ void dot( double *x, double *y, int n )
{
    const int tid = threadIdx.x;
    int i;
    double result = 0.0;
    for (i = tid; i < n; i += blockDim.x) {
        result += x[i] * y[i];
    }
    d_tmp[tid] = result;
}

void vec_init( double *x, double *y, int n )
{
    int i;
    const double tx[] = {1.0/64.0, 1.0/128.0, 1.0/256.0};
    const double ty[] = {1.0, 2.0, 4.0};
    const size_t arrlen = sizeof(tx)/sizeof(tx[0]);

    for (i=0; i<n; i++) {
        x[i] = tx[i % arrlen];
        y[i] = ty[i % arrlen];
    }
}

int main( int argc, char* argv[] )
{
    double *x, *y, *tmp, result;   /* host copies of x, y, tmp */
    double *d_x, *d_y;             /* device copies of x, y */
    int i, n = 1024*1024;
    const int max_len = 64 * n;
    const size_t size_tmp = BLKDIM * sizeof(*tmp);

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_len ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", max_len);
        return EXIT_FAILURE;
    }

    const size_t size = n*sizeof(*x);

    /* Allocate space for device copies of x, y */
    cudaSafeCall( cudaMalloc((void **)&d_x, size) );
    cudaSafeCall( cudaMalloc((void **)&d_y, size) );

    /* Allocate space for host copies of x, y */
    x = (double*)malloc(size);
    assert(x != NULL);
    y = (double*)malloc(size);
    assert(y != NULL);
    tmp = (double*)malloc(size_tmp);
    assert(tmp != NULL);
    vec_init(x, y, n);

    /* Copy inputs to device */
    cudaSafeCall( cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice) );

    /* Launch dot() kernel on GPU */
    printf("Computing the dot product of %d elements... ", n);
    dot<<<1, BLKDIM>>>(d_x, d_y, n);
    cudaCheckError();

    /* Copy result back to host */
    cudaSafeCall( cudaMemcpyFromSymbol(tmp, d_tmp, size_tmp) );

    /* Performs the last reduction on the CPU */
    result = 0.0;
    for (i=0; i<BLKDIM; i++) {
        result += tmp[i];
    }

    printf("result=%f\n", result);
    const double expected = ((double)n)/64;

    /* Check result */
    if ( fabs(result - expected) < 1e-5 ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED: got %f, expected %f\n", result, expected);
    }

    /* Cleanup */
    free(x);
    free(y);
    free(tmp);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_tmp);
    return EXIT_SUCCESS;
}
