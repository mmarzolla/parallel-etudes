/****************************************************************************
 *
 * cuda-dot-shared.cu - Dot product with CUDA using __shared__ memory
 *
 * Copyright (C) 2017--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-dot-shared.cu -o cuda-dot-shared -lm
 *
 * Run with:
 * ./cuda-dot-shared [len]
 *
 * Example:
 * ./cuda-dot-shared
 *
 ****************************************************************************/
#include <stdio.h>
#include <math.h>

#define BLKDIM 1024

__global__ void dot( double *x, double *y, int n, double *result )
{
    __shared__ double sums[BLKDIM];
    double local_sum = 0.0;

    const int tid = threadIdx.x;
    int i;

    for (i = tid; i < n; i += blockDim.x) {
        local_sum += x[i] * y[i];
    }
    sums[tid] = local_sum;
    __syncthreads(); /* Wait for all threads to write to the shared array */
    /* Thread 0 makes the final reduction */
    if ( 0 == tid ) {
        double sum = 0.0;
        for (i=0; i<blockDim.x; i++) {
            sum += sums[i];
        }
        *result = sum;
    }
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
    double *x, *y, result;              /* host copies of x, y, result */
    double *d_x, *d_y, *d_result;       /* device copies of x, y, result */
    int n = 1024*1024;
    const int max_len = 64 * n;

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

    const size_t size = n * sizeof(*x);

    /* Allocate space for device copies of x, y, result */
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&d_result, sizeof(*d_result));

    /* Allocate space for host copies of x, y */
    x = (double*)malloc(size);
    y = (double*)malloc(size);
    vec_init(x, y, n);

    /* Copy inputs to device */
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    /* Launch dot() kernel on GPU */
    printf("Computing the dot product of %d elements... ", n);
    dot<<<1, BLKDIM>>>(d_x, d_y, n, d_result);

    /* Copy result back to host */
    cudaMemcpy(&result, d_result, sizeof(*d_result), cudaMemcpyDeviceToHost);

    printf("result=%f\n", result);
    const double expected = ((double)n)/64;

    /* Check result */
    if ( fabs(result - expected) < 1e-5 ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED: got %f, expected %f\n", result, expected);
    }

    /* Cleanup */
    free(x); free(y);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_result);
    return EXIT_SUCCESS;
}
