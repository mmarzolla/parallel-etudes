/****************************************************************************
 *
 * omp-edge-detect.c - Edge detection on grayscale images
 *
 * Copyright (C) 2019, 2021, 2022 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Edge detection on grayscale images
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2022-08-19

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

The goal of this exercise is to parallelize the two nested loops in
the `edge_detect()` function using suitable OpenMP directives.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-edge-detect.c -o omp-edge-detect

To execute:

        ./omp-edge-detect [threshold] < input > output

Example:

        ./omp-edge-detect < BWstop-sign.pgm > BWstop-sign-edges.pgm

## Files

- [omp-edge-detect.c](omp-edge-detect.c)
- [BWstop-sign.pgm](BWstop-sign.pgm)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include "pgmutils.h"

/**
 * Edge detection using the Sobel operator
 */
void edge_detect( const PGM_image* in, PGM_image* edges, int threshold )
{
    const int width = in->width;
    const int height = in->height;
    /* The following C99 casts are used to convert `in` and `edges` to
       the type "array[width] of pointers to unsigned char", so that
       indexing can be done more easily */
    const unsigned char (*in_bmap)[width] = (const unsigned char (*)[width])(in->bmap);
    unsigned char (*edges_bmap)[width] = (unsigned char (*)[width])(edges->bmap);
#ifdef SERIAL
    /* [TODO] Parallelize the following loops */
#else
#pragma omp parallel for collapse(2) default(none) shared(in_bmap, edges_bmap, threshold, height, width, BLACK, WHITE)
#endif
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++)  {
            /* Compute the gradients Gx and Gy along the x and y
               dimensions */
            const int Gx =
                in_bmap[i-1][j-1] - in_bmap[i-1][j+1]
                + 2*in_bmap[i][j-1] - 2*in_bmap[i][j+1]
                + in_bmap[i+1][j-1] - in_bmap[i+1][j+1];
            const int Gy =
                in_bmap[i-1][j-1] + 2*in_bmap[i-1][j] + in_bmap[i-1][j+1]
                - in_bmap[i+1][j-1] - 2*in_bmap[i+1][j] - in_bmap[i+1][j+1];
            const int magnitude = Gx * Gx + Gy * Gy;
            if  (magnitude > threshold*threshold)
                edges_bmap[i][j] = WHITE;
            else
                edges_bmap[i][j] = BLACK;
        }
    }
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
    const double tstart = omp_get_wtime();
    edge_detect(&bmap, &out, threshold);
    const double elapsed = omp_get_wtime() - tstart;
    fprintf(stderr, "Execution time %f\n", elapsed);
    write_pgm(stdout, &out, "produced by omp-edge-detect.c");
    free_pgm(&bmap);
    free_pgm(&out);
    return EXIT_SUCCESS;
}
