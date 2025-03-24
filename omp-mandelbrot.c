/****************************************************************************
 *
 * omp-mandelbrot.c - Draw the Mandelbrot set
 *
 * Copyright (C) 2017--2025 Moreno Marzolla
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
% HPC - Mandelbrot set
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-03-24

![Benoit Mandelbrot (1924--2010)](Benoit_Mandelbrot.jpg)

The file [omp-mandelbrot.c](omp-mandelbrot.c) contains a serial
program that draws the Mandelbrot set. The Mandelbrot set is the set
of points $c$ in the complex plane such that the sequence:

$$
z_n = \begin{cases}
0 & \mbox{if $n = 0$}\\
z_{n-1}^2 + c & \mbox{if $n > 0$}
\end{cases}
$$

does not diverge as $n \rightarrow \infty$. This program uses a cutoff
value `MAXIT` so that, if the first `MAXIT` terms of the sequence
$z_n$ do not diverge, point $c$ is assumed to be part of the
Mandelbrot set.  For an intuitive explanation of the Mandelbrot set
you can use [this demo file](mandelbrot-set-demo.ggb) with
[GeoGebra](https://www.geogebra.org/calculator/).

The program accepts the vertical resolution of the output image as the
only optional parameter on the command line. The image is written to
the file `omp-mandelbrot.ppm` in PPM (_Portable Pixmap_) format. The
output image can be converted into a more familiar format, e.g., PNG,
using the `convert` utility of the
[ImageMagick](https://imagemagick.org/) package:

        convert omp-mandelbrot.ppm omp-mandelbrot.png

Modify the program to make use of shared-memory parallelism using
OpenMP. You should make sure that the result is exactly the same as
the serial program. To do so, compare the image produced by the serial
and parallel versions using the `cmp` command:

        cmp file1 file2

prints a message if the content `file1` and `file2` differ (the format
of the files is not important, since `cmp` performs a byte-by-byte
comparison).

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-mandelbrot.c -o omp-mandelbrot

Run with:

        ./omp-mandelbrot [ysize]

Example:

        ./omp-mandelbrot 800

## Files

- [omp-mandelbrot.c](omp-mandelbrot.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <omp.h>

const int MAXIT = 100;

typedef struct __attribute__((__packed__)) {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
const pixel_t COLORS[] = {
    { 66,  30,  15}, /* r, g, b */
    { 25,   7,  26},
    {  9,   1,  47},
    {  4,   4,  73},
    {  0,   7, 100},
    { 12,  44, 138},
    { 24,  82, 177},
    { 57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201,  95},
    {255, 170,   0},
    {204, 128,   0},
    {153,  87,   0},
    {106,  52,   3} };
const int NCOLORS = sizeof(COLORS)/sizeof(COLORS[0]);

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

/* Draw the Mandelbrot set on the bitmap pointed to by `p`. */
void draw_lines( pixel_t* bmap, int xsize, int ysize )
{
    const float XMIN = -2.3, XMAX = 1.0;
    const float SCALE = (XMAX - XMIN)*ysize / xsize;
    const float YMIN = -SCALE/2, YMAX = SCALE/2;

#ifndef SERIAL
#pragma omp parallel for schedule(dynamic) default(none) shared(bmap, xsize, ysize, MAXIT, NCOLORS, COLORS, XMIN, XMAX, YMIN, YMAX)
#endif
    for ( int y = 0; y < ysize; y++) {
        for ( int x = 0; x < xsize; x++ ) {
            pixel_t *p = &bmap[y*xsize + x];
            const float re = XMIN + (XMAX - XMIN) * (float)(x) / (xsize - 1);
            const float im = YMAX - (YMAX - YMIN) * (float)(y) / (ysize - 1);
            const int v = iterate(re, im);

            if (v < MAXIT) {
                p->r = COLORS[v % NCOLORS].r;
                p->g = COLORS[v % NCOLORS].g;
                p->b = COLORS[v % NCOLORS].b;
            } else {
                p->r = p->g = p->b = 0;
            }
        }
    }
}

int main( int argc, char *argv[] )
{
    FILE *out = NULL;
    const char* fname="omp-mandelbrot.ppm";
    pixel_t *bitmap = NULL;
    int xsize, ysize;

    if ( argc > 1 ) {
        ysize = atoi(argv[1]);
    } else {
        ysize = 1024;
    }

    xsize = ysize * 1.4;

    out = fopen(fname, "w");
    if ( !out ) {
        fprintf(stderr, "Error: cannot create %s\n", fname);
        return EXIT_FAILURE;
    }

    /* Write the header of the output file */
    fprintf(out, "P6\n");
    fprintf(out, "%d %d\n", xsize, ysize);
    fprintf(out, "255\n");

    /* Allocate the complete bitmap */
    bitmap = (pixel_t*)malloc(xsize*ysize*sizeof(*bitmap));
    assert(bitmap != NULL);
    const double tstart = omp_get_wtime();
    draw_lines(bitmap, xsize, ysize);
    const double elapsed = omp_get_wtime() - tstart;
    fwrite(bitmap, sizeof(*bitmap), xsize*ysize, out);
    fclose(out);
    free(bitmap);
    fprintf(stderr, "Elapsed time %f\n", elapsed);

    return EXIT_SUCCESS;
}
