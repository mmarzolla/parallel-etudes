/****************************************************************************
 *
 * cat-map-rectime.c - Minimum recurrence time of Arnold's  cat map
 *
 * Copyright (C) 2017--2021, 2024 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Minimum Recurrence Time of Arnold's cat map
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2024-10-04

This program computes the _Minimum Recurrence Time_ of Arnold's cat
map for an image of given size $N \times N$. The minimum recurrence
time is the minimum number of iterations of Arnold's cat map that
return back the original image.

The minimum recurrence time depends on the image size $n$, but no
simple relation is known. Table 1 shows the minimum recurrence time
for some values of $N$.

:Table 1: Minimum recurrence time for some image sizes $N$

    $N$   Minimum recurrence time
------- -------------------------
     64                        48
    128                        96
    256                       192
    512                       384
   1368                        36
------- -------------------------

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-cat-map-rectime.c -o omp-cat-map-rectime

Run with:

        ./omp-cat-map-rectime [N]

Example:

        ./omp-cat-map-rectime 1024

## Files

- [omp-cat-map-rectime.c](omp-cat-map-rectime.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

/* Compute the Greatest Common Divisor (GCD) of integers a>0 and b>0 */
int gcd(int a, int b)
{
    assert(a>0);
    assert(b>0);

    while ( b != a ) {
        if (a>b) {
            a = a-b;
        } else {
            b = b-a;
        }
    }
    return a;
}

/* compute the Least Common Multiple (LCM) of integers a>0 and b>0 */
int lcm(int a, int b)
{
    assert(a>0);
    assert(b>0);
    return (a / gcd(a, b))*b;
}

/**
 * Compute the recurrence time of Arnold's cat map applied to an image
 * of size (n*n). For each point (x,y), compute the minimum recurrence
 * time k(x,y). The minimum recurrence time for the whole image is the
 * Least Common Multiple of all k(x,y).
 */
int cat_map_rectime( int n )
{
#ifdef SERIAL
    /* [TODO] Implement this function; start with a working serial
       version, then parallelize. */
    return 0;
#else
    int x, y, rectime = 1;
    /* Since the inner body of the loops may require different time
       for different points, we apply a dynamic scheduling. */
#pragma omp parallel for collapse(2) schedule(dynamic, 32) default(none) shared(rectime, n)
    for (y=0; y<n; y++) {
        for (x=0; x<n; x++) {
            int xold = x, xnew;
            int yold = y, ynew;
            int k = 0;
            /* Iterate the cat map until (xnew, ynew) becomes equal to
               (x, y) */
            do {
                xnew = (2*xold+yold) % n;
                ynew = (xold + yold) % n;
                xold = xnew;
                yold = ynew;
                k++;
            } while (xnew != x || ynew != y);
            /* `k` is the minimum recurrence time of the pixel of
               coordinate (x,y). The minimum recurrence time of the
               whole image is the least common multiple of all values
               `k` computed for each pixel. */
#pragma omp critical
            rectime = lcm(rectime, k);
	}
    }
    return rectime;
#endif
}

int main( int argc, char* argv[] )
{
    int n, k;

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s image_size\n", argv[0]);
        return EXIT_FAILURE;
    }
    n = atoi(argv[1]);
    const double tstart = omp_get_wtime();
    k = cat_map_rectime(n);
    const double elapsed = omp_get_wtime() - tstart;
    printf("%d\n", k);

    printf("Elapsed time: %f\n", elapsed);

    return EXIT_SUCCESS;
}
