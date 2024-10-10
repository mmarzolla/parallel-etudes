/****************************************************************************
 *
 * opencl-dot-local.c - Dot product using __local memory
 *
 * Copyright (C) 2017--2021 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * cc opencl-dot-local.c simpleCL.c -o opencl-dot-local -lm -lOpenCL
 *
 * Run with:
 * ./opencl-dot-local [len]
 *
 * Example:
 * ./opencl-dot-local
 *
 ****************************************************************************/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "simpleCL.h"

void vec_init( float *x, float *y, int n )
{
    const float tx[] = {1.0/64.0, 1.0/128.0, 1.0/256.0};
    const float ty[] = {1.0, 2.0, 4.0};
    const size_t arrlen = sizeof(tx)/sizeof(tx[0]);

    for (int i=0; i<n; i++) {
        x[i] = tx[i % arrlen];
        y[i] = ty[i % arrlen];
    }
}

int main( int argc, char* argv[] )
{
    const float TOL = 1e-5;
    float *x, *y, result;               /* host copies of x, y, result */
    cl_mem d_x, d_y, d_result;          /* device copies of x, y, result */
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
    sclInitFromFile("opencl-dot-local.cl");
    sclKernel dot_kernel = sclCreateKernel("dot_kernel");

    /* Allocate space for host copies of x, y */
    x = (float*)malloc(size);
    y = (float*)malloc(size);
    vec_init(x, y, n);

    /* Allocate space for device copies of x, y, result */
    d_x = sclMallocCopy(size, x, CL_MEM_READ_ONLY);
    d_y = sclMallocCopy(size, y, CL_MEM_READ_ONLY);
    d_result = sclMalloc(sizeof(result), CL_MEM_WRITE_ONLY);

    /* Launch dot() kernel on the device */
    printf("Computing the dot product of %d elements... ", n);
    sclSetArgsEnqueueKernel(dot_kernel,
                            DIM1(sclRoundUp(n, SCL_DEFAULT_WG_SIZE)), DIM1(SCL_DEFAULT_WG_SIZE),
                            ":b :b :d :b",
                            d_x, d_y, n, d_result);

    /* Copy result back to host */
    sclMemcpyDeviceToHost(&result, d_result, sizeof(result));

    printf("result=%f\n", result);
    const float expected = ((float)n)/64;

    /* Check result */
    if ( fabsf(result - expected) < TOL ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED: got %f, expected %f\n", result, expected);
    }

    /* Cleanup */
    free(x); free(y);
    sclFree(d_x); sclFree(d_y); sclFree(d_result);
    sclFinalize();
    return EXIT_SUCCESS;
}
