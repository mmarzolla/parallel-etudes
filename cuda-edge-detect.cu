/****************************************************************************
 *
 * cuda-edge-detect.c - Edge detection on grayscale images
 *
 * Copyright 2024 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 ****************************************************************************/

/***
% HPC - Edge detection on grayscale images
% Moreno Marzolla <moreno.marzolla@unibo.i>
% Last updated: 2024-09-02

![Result of the Sobel operator](edge-detect.png)

The [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator) is
used to detect the edges on an grayscale image. The idea is to compute
the gradient of color change across each pixel; those pixels for which
the gradient exceeds a user-defined threshold are considered to be
part of an edge. Computation of the gradient involves the application
of a $3 \times 3$ stencil to the input image.

The program reads an input image fro standard input in
[PGM](https://en.wikipedia.org/wiki/Netpbm#PGM_example) (_Portable
Graymap_) format and produces a B/W image to standard output. The user
can specify an optional threshold on the command line.

The goal of this exercise is to parallelize the computation of the
Sobel operator using CUDA; this can be achieved by writing a kernel
that computes the edge at each pixel, and invoke the kernel from the
`edge_detect()` function.

To compile:

        nvcc cuda-edge-detect.cu -o cuda-edge-detect

To execute:

        ./cuda-edge-detect [threshold] < input > output

Example:

        ./cuda-edge-detect < BWstop-sign.pgm > BWstop-sign-edges.pgm

## Files

- [cuda-edge-detect.cu](cuda-edge-detect.cu)
- [BWstop-sign.pgm](BWstop-sign.pgm)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"
#include "pgmutils.h"

#ifndef SERIAL
#define BLKDIM 32

__device__ __host__
#endif
int IDX(int i, int j, int width)
{
    return (i*width + j);
}

#ifndef SERIAL
__global__ void
sobel_kernel(const unsigned char *in,
             unsigned char *edges,
             int width, int height,
             int threshold)
{
    const unsigned char WHITE = 255;
    const unsigned char BLACK = 0;
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int i = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    if (i==0 || j==0 || i==height-1 || j==width-1)
        edges[IDX(i, j, width)] = WHITE;
    else {
        /* Compute the gradients Gx and Gy along the x and y
           dimensions */
        const int Gx =
            in[IDX(i-1, j-1, width)] - in[IDX(i-1, j+1, width)]
            + 2*in[IDX(i, j-1, width)] - 2*in[IDX(i, j+1, width)]
            + in[IDX(i+1, j-1, width)] - in[IDX(i+1, j+1, width)];
        const int Gy =
            in[IDX(i-1, j-1, width)] + 2*in[IDX(i-1, j, width)] + in[IDX(i-1, j+1, width)]
            - in[IDX(i+1, j-1, width)] - 2*in[IDX(i+1, j, width)] - in[IDX(i+1, j+1, width)];
        const int magnitude = Gx * Gx + Gy * Gy;
        if  (magnitude > threshold*threshold)
            edges[IDX(i, j, width)] = WHITE;
        else
            edges[IDX(i, j, width)] = BLACK;
    }
}
#endif

/**
 * Edge detection using the Sobel operator
 */
void edge_detect( const PGM_image* in, PGM_image* edges, int threshold )
{
    const int width = in->width;
    const int height = in->height;
#ifdef SERIAL
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++)  {
            /* Compute the gradients Gx and Gy along the x and y
               dimensions */
            const int Gx =
                in->bmap[IDX(i-1, j-1, width)] - in->bmap[IDX(i-1, j+1, width)]
                + 2*in->bmap[IDX(i, j-1, width)] - 2*in->bmap[IDX(i, j+1, width)]
                + in->bmap[IDX(i+1, j-1, width)] - in->bmap[IDX(i+1, j+1, width)];
            const int Gy =
                in->bmap[IDX(i-1, j-1, width)] + 2*in->bmap[IDX(i-1, j, width)] + in->bmap[IDX(i-1, j+1, width)]
                - in->bmap[IDX(i+1, j-1, width)] - 2*in->bmap[IDX(i+1, j, width)] - in->bmap[IDX(i+1, j+1, width)];
            const int magnitude = Gx * Gx + Gy * Gy;
            if  (magnitude > threshold*threshold)
                edges->bmap[IDX(i, j, width)] = WHITE;
            else
                edges->bmap[IDX(i, j, width)] = BLACK;
        }
    }
#else
    const size_t size = width * height;
    const dim3 block(BLKDIM, BLKDIM);
    const dim3 grid((width + BLKDIM-1)/BLKDIM, (height + BLKDIM-1)/BLKDIM);
    unsigned char *d_in, *d_edges;
    cudaSafeCall( cudaMalloc((void**)&d_in, size) );
    cudaSafeCall( cudaMalloc((void**)&d_edges, size) );
    cudaSafeCall( cudaMemcpy(d_in, in->bmap, size, cudaMemcpyHostToDevice) );
    sobel_kernel<<< grid, block >>>(d_in, d_edges, width, height, threshold);
    cudaSafeCall( cudaMemcpy(edges->bmap, d_edges, size, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaFree(d_in) );
    cudaSafeCall( cudaFree(d_edges) );
#endif
}

int main( int argc, char* argv[] )
{
    PGM_image bmap, out;
    int threshold = 70;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [threshold] < in.pgm > out.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }
    if ( argc > 1 ) {
        threshold = atoi(argv[1]);
    }
    read_pgm(stdin, &bmap);
    init_pgm(&out, bmap.width, bmap.height, WHITE);
    const double tstart = hpc_gettime();
    edge_detect(&bmap, &out, threshold);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Execution time %f\n", elapsed);
    write_pgm(stdout, &out, "produced by opencl-edge-detect.c");
    free_pgm(&bmap);
    free_pgm(&out);
    return EXIT_SUCCESS;
}