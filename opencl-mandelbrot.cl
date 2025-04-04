/****************************************************************************
 *
 * opencl-mandelbrot.cl - Kernel for opencl-mandelbrot.c
 *
 * Copyright (C) 2022 Moreno Marzolla
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

int __constant MAXIT = 100;

typedef uchar uint8_t;

typedef struct {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
pixel_t __constant colors[] = {
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
int __constant NCOLORS = sizeof(colors)/sizeof(colors[0]);

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
    int it;
    float x = 0.0f, y = 0.0f, xnew, ynew;
    for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0f*2.0f); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0f*x*y + cy;
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
__kernel void
mandelbrot_kernel( int xsize, int ysize, __global pixel_t* bmap)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < xsize && y < ysize) {
        __global pixel_t *p = bmap + y * xsize + x;
        const float XMIN = -2.3, XMAX = 1.0;
        const float SCALE = (XMAX - XMIN)*ysize / xsize;
        const float YMIN = -SCALE/2, YMAX = SCALE/2;
        const float re = XMIN + (XMAX - XMIN) * (float)(x) / (xsize - 1);
        const float im = YMAX - (YMAX - YMIN) * (float)(y) / (ysize - 1);
        const int v = iterate(re, im);

        if (v < MAXIT) {
            p->r = colors[v % NCOLORS].r;
            p->g = colors[v % NCOLORS].g;
            p->b = colors[v % NCOLORS].b;
        } else {
            p->r = p->g = p->b = 0;
        }
    }
}
