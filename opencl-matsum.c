/****************************************************************************
 *
 * opencl-matsum.c - Matrix-matrix addition
 *
 * Copyright (C) 2017--2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last update: 2024-01-04

The program [opencl-matsum.c](opencl-matsum.c) computes the sum of two
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
it nos an integer multiple of the workgroup size. Note that there is
no need to use local memory (why?).

To compile:

        cc -std=c99 -Wall -Wpedantic opencl-matsum.c simpleCL.c -o opencl-matsum -lm -lOpenCL

To execute:

        ./opencl-matsum [N]

Example:

        ./opencl-matsum 1024
## Files

- [opencl-matsum.c](opencl-matsum.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

***/

/* The following #define is required by the implementation of
   hpc_gettime(). It MUST be defined before including any other
   file. */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
#include "hpc.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "simpleCL.h"

#ifndef SERIAL
const char *program =
    "__kernel void matsum_kernel( __global const float *p,\n"
    "                             __global const float *q,\n"
    "                             __global float *r,\n"
    "                             int n )\n"
    "{\n"
    "    const int i = get_global_id(1);\n"
    "    const int j = get_global_id(0);\n"
    "    if ( i<n && j<n )\n"
    "        r[i*n + j] = p[i*n + j] + q[i*n + j];\n"
    "}\n";
sclKernel matsum_kernel;
#endif

void matsum( float *p, float *q, float *r, int n )
{
#ifdef SERIAL
    /* [TODO] Modify the body of this function to
       - allocate memory on the device
       - copy p and q to the device
       - call an appropriate kernel
       - copy the result back from the device to the host
       - free memory on the device
    */
    int i, j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            r[i*n + j] = p[i*n + j] + q[i*n + j];
        }
    }
#else
    const size_t size = n*n*sizeof(*p);
    const sclDim block = DIM2(SCL_DEFAULT_WG_SIZE2D, SCL_DEFAULT_WG_SIZE2D);
    const sclDim grid = DIM2(sclRoundUp(n, SCL_DEFAULT_WG_SIZE2D),
                             sclRoundUp(n, SCL_DEFAULT_WG_SIZE2D));

    /* Allocate space for device copies of p, q, r */
    cl_mem d_p = sclMallocCopy(size, p, CL_MEM_READ_ONLY);
    cl_mem d_q = sclMallocCopy(size, q, CL_MEM_READ_ONLY);
    cl_mem d_r = sclMalloc(size, CL_MEM_WRITE_ONLY);

    /* Launch matsum() kernel on GPU */
    sclSetArgsEnqueueKernel(matsum_kernel,
                            grid, block,
                            ":b :b :b :d",
                            d_p, d_q, d_r, n);

    /* Copy result back to host */
    sclMemcpyDeviceToHost(r, d_r, size);

    sclFree(d_p); sclFree(d_q); sclFree(d_r);
#endif
}

/* Initialize square matrix p of size nxn */
void fill( float *p, int n )
{
    int i, j, k=0;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            p[i*n+j] = k;
            k = (k+1) % 1000;
        }
    }
}

/* Check result */
int check( float *r, int n )
{
    const float TOL = 1e-5;
    int i, j, k = 0;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (fabsf(r[i*n+j] - 2.0f*k) > TOL) {
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

#ifndef SERIAL
    sclInitFromString(program);
    matsum_kernel = sclCreateKernel("matsum_kernel");
#endif

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

    printf("Elapsed time: %f\n", elapsed);

    /* Check result */
    check(r, n);

    /* Cleanup */
    free(p); free(q); free(r);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
