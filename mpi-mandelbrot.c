/****************************************************************************
 *
 * mpi-mandelbrot.c - Draw the Mandelbrot set with MPI
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Draw the Mandelbrot set with MPI
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-02

![Benoit Mandelbrot (1924--2010)](Benoit_Mandelbrot.jpg)

The file [mpi-mandelbrot.c](mpi-mandelbrot.c) contains the skeleton of
an MPI program that computes the Mandelbrot set; it is not a real
parallel version, since the master process does everything.

The program accepts the image height as an optional command-line
parameter; the image width is computed automatically to include the
whole set. The program writes a graphical representation of the
Mandelbrot set into a file `mandebrot.ppm` in PPM (_Portable Pixmap_)
format. If you don't have a suitable viewer, you can convert the
image, e.g., into PNG with the command:

        convert mandelbrot.ppm mandelbrot.png

The goal of this exercise is to write a truly parallel implementation
of the program, where all MPI processes contribute to the
computation. To do this, partition the image into $P$ vertical blocks
where $P$ is the number of MPI processes, and let each process compute
a portion of the image (see Figure 1).

![Figure 1: Domain decomposition for the computation of the Mandelbrot
 set with 4 MPI processes](mpi-mandelbrot.png)

Specifically, each process computes a portion of the image of size
$\mathit{xsize} \times (\mathit{ysize} / P)$. This is an
_embarrassingly parallel_ computation, since there is no need to
communicate. However, the processes _do_ need to send their portion of
image to the master, which will take care of stitching them together
to form the final result. This is done using the `MPI_Gather()`
function. This is a color image where three bytes are used to encode
the color of each pixel. Therefore, the `MPI_Gather()` operation will
transfer blocks of $(3 \times \mathit{xsize} \times \mathit{ysize} /
P)$ elements of type `MPI_BYTE`.

You can initially assume that the vertical size _ysize_ is an integer
multiple of $P$, and then relax this assumption. To this aim, you may
let process 0 take care of computing the last `(ysize % P)` rows of
the image, or use `MPI_Gatherv()` to allow different block sizes to be
assembled together.

You may want to keep the serial program as a reference§; to check the
correctness of the parallel implementation, you can compare the output
images produced by both versions with the command:

        cmp file1 file2

Both images should be identical; if not, something is wrong.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-mandelbrot.c -o mpi-mandelbrot

To execute:

        mpirun -n NPROC ./mpi-mandelbrot [ysize]

Example:

        mpirun -n 4 ./mpi-mandelbrot 800

## Files

- [mpi-mandelbrot.c](mpi-mandelbrot.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <mpi.h>

const int MAXIT = 100;

/* The __attribute__(( ... )) definition is gcc-specific, and tells
   the compiler that the fields of this structure should not be padded
   or aligned in any way. */
typedef struct __attribute__((__packed__)) {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
const pixel_t colors[] = {
    {66, 30, 15}, /* r, g, b */
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
    {106, 52, 3} };
const int NCOLORS = sizeof(colors)/sizeof(colors[0]);

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first `n` such that `z_n > bound`, or `MAXIT` if `z_n` is below
 * `bound` after `MAXIT` iterations.
 */
int iterate( float cx, float cy )
{
    float x = 0.0f, y = 0.0f, xnew, ynew;
    int it;
    for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0*2.0); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0*x*y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

/* Draw the rows of the Mandelbrot set from `ystart` (inclusive) to
   `yend` (excluded) to the bitmap pointed to by `p`. Note that `p`
   must point to the beginning of the bitmap where the portion of
   image will be stored; in other words, this function writes to
   pixels p[0], p[1], ... `xsize` and `ysize` MUST be the sizes
   of the WHOLE image. */
void draw_lines( int ystart, int yend, pixel_t* p, int xsize, int ysize )
{
    int x, y;
    for ( y = ystart; y < yend; y++) {
        for ( x = 0; x < xsize; x++ ) {
            const float cx = -2.5 + 3.5 * (float)x / (xsize - 1);
            const float cy = 1 - 2.0 * (float)y / (ysize - 1);
            const int v = iterate(cx, cy);
            if (v < MAXIT) {
                p->r = colors[v % NCOLORS].r;
                p->g = colors[v % NCOLORS].g;
                p->b = colors[v % NCOLORS].b;
            } else {
                p->r = p->g = p->b = 0;
            }
            p++;
        }
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    FILE *out = NULL;
    const char* fname="mpi-mandelbrot.ppm";
    pixel_t *bitmap = NULL;
    int xsize, ysize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        ysize = atoi(argv[1]);
    } else {
        ysize = 1024;
    }

    xsize = ysize * 1.4;

    /* xsize and ysize are known to all processes */
    if ( 0 == my_rank ) {
        out = fopen(fname, "w");
        if ( !out ) {
            fprintf(stderr, "Error: cannot create %s\n", fname);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Write the header of the output file */
        fprintf(out, "P6\n");
        fprintf(out, "%d %d\n", xsize, ysize);
        fprintf(out, "255\n");

        /* Allocate the complete bitmap */
        bitmap = (pixel_t*)malloc(xsize*ysize*sizeof(*bitmap));
        assert(bitmap != NULL);
#ifdef SERIAL
        /* [TODO] This is not a true parallel version, since the master
           does everything */
        draw_lines(0, ysize, bitmap, xsize, ysize);
        fwrite(bitmap, sizeof(*bitmap), xsize*ysize, out);
        fclose(out);
        free(bitmap);
#endif
    }

#ifndef SERIAL
#ifdef USE_GATHERV
    /* This version makes use of MPI_Gatherv to collect portions of
       different sizes. To compile this version, use:

       mpicc -std=c99 -Wall -Wpedantic -DUSE_GATHERV mpi-mandelbrot.c -o mpi-mandelbrot

    */
    int ystart[comm_sz], yend[comm_sz], counts[comm_sz], displs[comm_sz];
    for (int i=0; i<comm_sz; i++) {
        ystart[i] = ysize * i / comm_sz;
        yend[i] = ysize * (i+1) / comm_sz;
        /* counts[] and displs[] must be measured as the number of
           "array elements", NOT bytes; however, in this case the type
           of array elements that are gathered together is MPI_BYTE
           (see MPI_Gatherv below), so we need to multiply by
           sizeof(pixel_t) */
        counts[i] = (yend[i] - ystart[i])*xsize*sizeof(pixel_t);
        displs[i] = ystart[i]*xsize*sizeof(pixel_t);
    }

    pixel_t *local_bitmap = (pixel_t*)malloc(counts[my_rank]);
    assert(local_bitmap != NULL);

    const double tstart = MPI_Wtime();

    draw_lines( ystart[my_rank], yend[my_rank], local_bitmap, xsize, ysize);

    MPI_Gatherv( local_bitmap,          /* sendbuf      */
                 counts[my_rank],       /* sendcount    */
                 MPI_BYTE,              /* datatype     */
                 bitmap,                /* recvbuf      */
                 counts,                /* recvcounts[] */
                 displs,                /* displacements[] */
                 MPI_BYTE,              /* datatype     */
                 0,                     /* root         */
                 MPI_COMM_WORLD
                 );

    const double elapsed = MPI_Wtime() - tstart;

    if ( 0 == my_rank ) {
        fwrite(bitmap, sizeof(*bitmap), xsize*ysize, out);
        fclose(out);

        printf("Elapsed time (s): %f\n", elapsed);
    }
    free(bitmap);
    free(local_bitmap);
#else
    const int local_ysize = ysize / comm_sz;
    const int ystart = local_ysize * my_rank;
    const int yend = local_ysize * (my_rank + 1);
    pixel_t *local_bitmap = (pixel_t*)malloc(xsize*local_ysize*sizeof(*local_bitmap));
    assert(local_bitmap != NULL);

    const double tstart = MPI_Wtime();

    draw_lines( ystart, yend, local_bitmap, xsize, ysize);

    MPI_Gather( local_bitmap,           /* sendbuf      */
                xsize*local_ysize*3,    /* sendcount    */
                MPI_BYTE,               /* datatype     */
                bitmap,                 /* recvbuf      */
                xsize*local_ysize*3,    /* recvcount    */
                MPI_BYTE,               /* datatype     */
                0,                      /* root         */
                MPI_COMM_WORLD
                );

    if ( 0 == my_rank ) {
        /* the master computes the last (ysize % comm_sz) lines of the image */
        if ( ysize % comm_sz) {
            const int skip = local_ysize * comm_sz; /* how many rows to skip */
            draw_lines( skip, ysize,
                        &bitmap[skip*xsize],
                        xsize, ysize );
        }
        const double elapsed = MPI_Wtime() - tstart;

        fwrite(bitmap, sizeof(*bitmap), xsize*ysize, out);
        fclose(out);

        printf("Elapsed time (s): %f\n", elapsed);
    }
    free(bitmap);
    free(local_bitmap);
#endif
#endif

    MPI_Finalize();

    return EXIT_SUCCESS;
}
