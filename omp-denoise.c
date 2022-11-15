/****************************************************************************
 *
 * omp-denoise.c - Image denoising
 *
 * Copyright 2018--2022 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Image denoising
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-08-16

![Figure 1: Denoising example (original image by Simpsons, CC BY-SA 3.0, <https://commons.wikimedia.org/w/index.php?curid=8904364>)](denoise.png)

The file [omp-denoise.c](omp-denoise.c) contains a serial
implementation of an _image denoising_ algorithm that (to some extent)
can be used to "cleanup" color images. The algorithm replaces the
color of each pixel with the _median_ of the four adjacent pixels plus
itself (_median-of-five_).  The median-of-five algorithm is applied
separately for each color channel (red, green, and blue).

This is particularly useful for removing "hot pixels", i.e., pixels
whose color is way off its intended value, for example due to problems
in the sensor used to acquire the image. However, depending on the
amount of noise, a single pass could be insufficient to remove every
hot pixel; see Figure 1.

The input image is read from standard input in
[PPM](http://netpbm.sourceforge.net/doc/ppm.html) (Portable Pixmap)
format; the result is written to standard output in the same format.q

To compile:

        gcc -std=c99 -fopenmp -Wall -Wpedantic omp-denoise.c -o omp-denoise

To execute:

        ./omp-denoise < input > output

Example:

        OMP_NUM_THREADS=4 ./omp-denoise < giornale.ppm > giornale-denoised.ppm

## Files

- [omp-denoise.c](omp-denoise.c)
- [giornale.ppm](giornale.ppm) (sample input)

 ***/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include "ppmutils.h"

/**
 * Swap *a and *b if necessary so that, at the end, *a <= *b
 */
void compare_and_swap( unsigned char *a, unsigned char *b )
{
    if (*a > *b ) {
        unsigned char tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

int IDX(int i, int j, int width)
{
    return (i*width + j);
}

/**
 * Return the median of v[0..4]; upon termination, the input vector
 * could be modified (permuted).
 */
unsigned char median_of_five( unsigned char v[5] )
{
    /* We do a partial sort of v[5] using bubble sort until v[2] is
       correctly placed; this element is the median. (There are better
       ways to compute the median-of-five). */
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v  , v+1 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    return v[2];
}

/**
 * Denoise a single color channel
 */
void denoise( unsigned char *bmap, int width, int height )
{
    unsigned char *out = (unsigned char*)malloc(width*height);
    unsigned char v[5];
    assert(out != NULL);

    memcpy(out, bmap, width*height);
#ifndef SERIAL
#pragma omp parallel for collapse(2) default(none) shared(width, height, bmap, out) private(v)
#endif
    for (int i=1; i<height - 1; i++) {
        for (int j=1; j<width - 1; j++) {
            const int CENTER = IDX(i, j, width);
            const int LEFT = IDX(i, j-1, width);
            const int RIGHT = IDX(i, j+1, width);
            const int TOP = IDX(i-1, j, width);
            const int BOTTOM = IDX(i+1, j, width);

            v[0] = bmap[CENTER];
            v[1] = bmap[LEFT];
            v[2] = bmap[RIGHT];
            v[3] = bmap[TOP];
            v[4] = bmap[BOTTOM];

            out[CENTER] = median_of_five(v);
        }
    }
    memcpy(bmap, out, width*height);
    free(out);
}

int main( void )
{
    PPM_image img;
    read_ppm(stdin, &img);
    const double tstart = omp_get_wtime();
    denoise(img.r, img.width, img.height);
    denoise(img.g, img.width, img.height);
    denoise(img.b, img.width, img.height);
    const double elapsed = omp_get_wtime() - tstart;
    fprintf(stderr, "Execution time: %f\n", elapsed);
    write_ppm(stdout, &img, "produced by omp-denoise.c");
    free_ppm(&img);
    return EXIT_SUCCESS;
}
