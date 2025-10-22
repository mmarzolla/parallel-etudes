/****************************************************************************
 *
 * omp-xcollapse.c - simulate the "collapse()" clause
 *
 * Copyright (C) 2025 Moreno Marzolla
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
% Simulate the "collapse()" clause
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-22

In this exercise we simulate the `collapse()` clause by hand; in other
words, we want to replace the statement:

```
#pragma omp parallel for collapse(2)
for (int i=...) {
  for (int j=...) {
    f(i,j);
  }
}
```

with a single parallel loop:

```
#pragma omp parallel for
for (int idx=...) {
  const int i = ...
  const int j = ...
  f(i,j);
}
```

where indices `i` and `j` are derived from `idx`. The goal is to
understand how it is possible to encode the index space of multiple
nested loops into a single index variable `idx`, and "unpack" the
index variables of the original loops from `idx`.

The function `erode(in, out)` applies the morphological operator
_erosion_ to image `in`, and writes the result into the output image
`out`. The erosion operator replaces the color of each pixel with the
maximum color of the neighborhood of size $3 \times 3$.

If the input image is B/W, where white pixels are encoded with high
values and black pixels with low values, the erosion operator grows
white areas. This program applies erosion many times, according to a
command-line parameter.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-collapse.c -o omp-collapse

To execute:

        ./omp-collapse [niter] < in.pgm > out.pgm

Example:

        OMP_NUM_THREADS=2 ./omp-collapse 10 < cat1024.pgm > out.pgm

## Files

- [omp-collapse.c](omp-collapse.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "pgmutils.h"

unsigned char max9(const unsigned char v[9])
{
    unsigned char result = v[0];
    for (int i=1; i<9; i++) {
        if (result < v[i])
            result = v[i];
    }
    return result;
}

/**
 * Edge detection using the Sobel operator
 */
void erode( const PGM_image* in, PGM_image* out )
{
    const int width = in->width;
    const int height = in->height;

#ifdef SERIAL
#pragma omp parallel for collapse(2) default(none) shared(height, width, in, out)
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++)  {
            const int CENTER = i * width + j;
            const int NORTH = CENTER - width;
            const int SOUTH = CENTER + width;
            const int EAST  = CENTER - 1;
            const int WEST  = CENTER + 1;
            const int NE    = NORTH - 1;
            const int NW    = NORTH + 1;
            const int SE    = SOUTH - 1;
            const int SW    = SOUTH + 1;
            const unsigned char v[] = {in->bmap[CENTER],
                                       in->bmap[NORTH],
                                       in->bmap[SOUTH],
                                       in->bmap[EAST],
                                       in->bmap[WEST],
                                       in->bmap[NE],
                                       in->bmap[NW],
                                       in->bmap[SE],
                                       in->bmap[SW]};
            out->bmap[CENTER] = max9(v);
        }
    }
#else
    #pragma omp parallel for default(none) shared(height, width, in, out)
    for (int idx = 0; idx < (height-1)*(width-1); idx++) {
        int tmp = idx;
        const int j = 1 + tmp % width;
        tmp /= width;
        const int i = 1 + tmp;

        const int CENTER = i * width + j;
        const int NORTH = CENTER - width;
        const int SOUTH = CENTER + width;
        const int EAST  = CENTER - 1;
        const int WEST  = CENTER + 1;
        const int NE    = NORTH - 1;
        const int NW    = NORTH + 1;
        const int SE    = SOUTH - 1;
        const int SW    = SOUTH + 1;
        const unsigned char v[] = {in->bmap[CENTER],
                                   in->bmap[NORTH],
                                   in->bmap[SOUTH],
                                   in->bmap[EAST],
                                   in->bmap[WEST],
                                   in->bmap[NE],
                                   in->bmap[NW],
                                   in->bmap[SE],
                                   in->bmap[SW]};
        out->bmap[CENTER] = max9(v);
    }
#endif
}

int main( int argc, char* argv[] )
{
    PGM_image img1, img2;
    PGM_image *cur, *next;
    int niter = 5;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n] < in.pgm > out.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        niter = atoi(argv[1]);
    }

    read_pgm(stdin, &img1);
    init_pgm(&img2, img1.width, img1.height, WHITE);
    cur = &img1;
    next = &img2;
    const double tstart = omp_get_wtime();
    for (int i=0; i<niter; i++) {
        erode(cur, next);
        PGM_image *tmp = cur;
        cur = next;
        next = tmp;
    }
    const double elapsed = omp_get_wtime() - tstart;
    fprintf(stderr, "Execution time %.3f\n", elapsed);
    write_pgm(stdout, cur, "produced by omp-collapse.c");
    free_pgm(cur);
    free_pgm(next);
    return EXIT_SUCCESS;
}
