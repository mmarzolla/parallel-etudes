/****************************************************************************
 *
 * omp-cat-map.c - Arnold's cat map
 *
 * Copyright (C) 2016--2023 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Arnold's cat map
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-11-03

![](cat-map.png)

[Arnold's cat map](https://en.wikipedia.org/wiki/Arnold%27s_cat_map)
is a continuous chaotic function that has been studied in the '60s by
the Russian mathematician [Vladimir Igorevich
Arnold](https://en.wikipedia.org/wiki/Vladimir_Arnold). In its
discrete version, the function can be understood as a transformation
of a bitmapped image $P$ of size $N \times N$ into a new image $P'$ of
the same size. For each $0 \leq x, y < N$, the pixel of coordinates
$(x,y)$ in $P$ is mapped into a new position $C(x, y) = (x', y')$ in
$P'$ such that

$$
x' = (2x + y) \bmod N, \qquad y' = (x + y) \bmod N
$$

("mod" is the integer remainder operator, i.e., operator `%` of the C
language). We may assume that $(0, 0)$ is top left and $(N-1, N-1)$ is
bottom right, so that the bitmap can be encoded as a regular
two-dimensional C matrix.

The transformation corresponds to a linear "stretching" of the image,
that is then broken down into triangles that are rearranged as shown
in Figure 1.

![Figure 1: Arnold's cat map](cat-map.svg)

Arnold's cat map has interesting properties. Let $C^k(x, y)$ be the
$k$-th iterate of $C$, i.e.:

$$
C^k(x, y) = \begin{cases}
(x, y) & \mbox{if $k=0$}\\
C(C^{k-1}(x,y)) & \mbox{if $k>0$}
\end{cases}
$$

Therefore, $C^2(x,y) = C(C(x,y))$, $C^3(x,y) = C(C(C(x,y)))$, and so
on.

If we apply $C$ once, we get a severely distorted version of the
input. If we apply $C$ on the result, we get an even more distorted
image. As we keep applying $C$, the original image is no longer
discernible. However, after a certain number of iterations, that
depends on the image size $N$ and has been proved to never exceed
$3N$, we get back the original image! (Figure 2).

![Figure 2: Some iterations of the cat map](cat-map-demo.png)

The _minimum recurrence time_ for an image is the minimum positive
integer $k \geq 1$ such that $C^k(x, y) = (x, y)$ for all $(x, y)$.
The minimum recurrence time is the minimum number of iterations of the
cat map that produce the starting image. For example, the minimum
recurrence time for [cat1368.pgm](cat1368.pgm) of size $1368 \times
1368$ is $36$.

The minimum recurrence time depends on the image size $N$. No closed
formula is known to compute the minimum recurrence time given the
image size $N$, although there are results and bounds that apply to
specific cases.

You are given a serial program that computes the $k$-th iterate of
Arnold's cat map on a square image. The program reads the input from
standard input in [PGM](https://en.wikipedia.org/wiki/Netpbm)
(_Portable GrayMap_) format. The results is printed to standard output
in PGM format. For example:

        ./omp-cat-map 100 < cat1368.pgm > cat1368-100.pgm

applies the cat map $k=100$ times on `cat1368.phm` and saves the
result to `cat1368-100.pgm`.

To display a PGM image you might need to convert it to a different
format, e.g., JPEG. Under Linux you can use `convert` from the
[ImageMagick](https://imagemagick.org/) package:

        convert cat1368-100.pgm cat1368-100.jpeg

Modify the function `cat_map()` to make use of shared-memory
parallelism using OpenMP. You might want to take advantage from the
fact that Arnold's cat map is _invertible_, and this implies that any
two different points $(x_1, y_1)$ and $(x_2, y_2)$ are always mapped
to different points $(x'_1, y'_1) = C(x_1, y_1)$ and $(x'_2, y'_2) =
C(x_2, y_2)$. Therefore, the output image $P'$ can be filled up in
parallel without race conditions (however, see below for some
caveats).

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-cat-map.c -o omp-cat-map

To execute:

        ./omp-cat-map k < input_file > output_file

Example:

        ./omp-cat-map 100 < cat1368.pgm > cat1368-100.pgm

## Suggestions

The provided function `cat_map()` is based on the following template:

```C
for (i=0; i<k; i++) {
        for (y=0; y<N; y++) {
                for (x=0; x<N; x++) {
                        (x', y') = C(x, y);
                        P'(x', y') = P(x, y);
                }
        }
        P = P';
}
```

The two inner loops build $P'$ from $P$; the outer loop applies this
transformation $k$ times, using the result of the previous iteration
as the source image. Therefore, the outer loop can _not_ be
parallelized (the result of an iteration is used as input for the next
one). Therefore, in the version above you can either:

1. Parallelize the `y` loop only, or

2. Parallelize the `x` loop only, or

3. Parallelize both the `y` and `x` loops using the `collapse(2)`
   clause.

(I suggest to try options 1 and/or 3. Option 2 does not appear to be
efficient in practice: why?).

We can apply the _loop interchange_ transformation to rewrite the
code above as follows:

```C
for (y=0; y<N; y++) {
        for (x=0; x<N; x++) {
                xcur = x; ycur = y;
                for (i=0; i<k; i++) {
                        (xnext, ynext) = C(xcur, ycur);
                        xcur = xnext;
                        ycur = ynext;
                }
                P'(xnext, ynext) = P(x, y)
        }
}
```

This version can be understood as follows: the two outer loops iterate
over all pixels $(x, y)$. For each pixel, the inner loop computes the
target position $(\mathit{xnext}, \mathit{ynext}) = C^k(x,y)$ that the
pixel of coordinates $(x, y)$ will occupy after $k$ iterations of the
cat map.

In this second version, we have the following options:

a. Parallelize the outermost loop on `y`, or

b. Parallelize the middle loop on `x`, or

c. Parallelize the two outermost loops with the `collapse(2)` directive.

(I suggest to try option c).

Intuitively, we might expect that (c) performs better than (3), because:

- the loop granularity is higher, and

- there are fewer writes to memory.

Interestingly, this does not appear to be the case (at least, not on
every processor). Table 1 shows the execution time of two versions of
the `cat_map()` function ("No loop interchange" refers to option (3);
"Loop interchange" refers to option (c)). The program has been
compiled with:

        gcc -O0 -fopenmp omp-cat-map.c -o omp-cat-map

(`-O0` prevents the compiler fro making code transformations that
might alter the functions too much) and executed as:

        ./omp-cat-map 2048 < cat1368.pgm > /dev/null

Each measurement is the average of five independent executions.

:Table 1: Execution time (in seconds) of the command `./omp-cat-map 2048 < cat1368.pgm > /dev/null` using all processor cores, with different implementations of the cat map iteration.

Processor           Cores   GHz  GCC version  No loop interchange   Loop interchange
------------------ ------ ----- ------------ -------------------- ------------------
Intel Xeon E3-1220      4   3.5       11.4.0                 6.84              12.90
Intel Xeon E5-2603     12   1.7        9.4.0                 6.11               7.74
Intel i7-4790         4+4   3.6        9.4.0                 6.05               5.89
Intel i7-9800X        8+8   3.8       11.4.0                 2.25               2.34
Intel i5-11320H       4+4   4.5        9.4.0                 3.94               4.01
Intel Atom N570       2+2   1.6        7.5.0               128.69              92.47
Raspberry Pi 4          4   1.5        8.3.0                27.10              27.24

On some platforms (Intel i5, i7 and Raspberry Pi 4) there is little or
no difference between the two versions. Loop interchange provides a
significant performance boost on the very old Intel Atom N570
processor, but provides worse performance on the Xeon processors.

## To probe further

What is the minimum recurrence time of image
[cat1024.pgm](cat1024.pgm) of size $1024 \times 1024$?  To answer this
question we need to iterate the cat map and stop as soon as we get
back the initial image.

It turns out that there is a smarter way, that does not even require
an input image but only its size $N$. To understand how it works, let
us suppose that we know that one particular pixel of the image, say
$(x_1, y_1)$, has minimum recurrence time 15. This means that after 15
iterations, the pixel at coordinates $(x_1, y_1)$ will return to its
starting position. Suppose that another pixel of coordinates $(x_2,
y_2)$ has minimum recurrence time 21. How many iterations of the cat
map are required to have _both_ pixels back to their original
positions?

The answer is $105$, which is the least common multiple (LCM) of 15
and 21. From this observation we can devise the following algorithmn
for computing the minimum recurrence time of an image of size $N
\times N$. Let $T(x,y)$ be the minimum recurrence time of the pixel of
coordinates $(x, y)$, $0 \leq x, y < N$. Then, the minimum recurrence
time of the whole image is the least common multiple of all $T(x, y)$.

[omp-cat-map-rectime.c](omp-cat-map-rectime.c) contains an incomplete
skeleton of a program that computes the minimum recurrence time of a
square image of size $N \times N$. Complete the program and then
produce a parallel version using the appropriate OpenMP directives.

Table 2 shows the minimum recurrence time for some $N$.

:Table 2: Minimum recurrence time for some image sizes $N$

    $N$   Minimum recurrence time
------- -------------------------
     64                        48
    128                        96
    256                       192
    512                       384
   1368                        36
------- -------------------------

Figure 3 shows the minimum recurrence time as a function of
$N$. Despite the fact that the actual values jump, there is a clear
tendency to align along straight lines.

![Figure 3: Minimum recurrence time as a function of the image size $N$](cat-map-rectime.png)

## Files

- [omp-cat-map.c](omp-cat-map.c)
- [omp-cat-map-rectime.c](omp-cat-map-rectime.c)
- [cat1024.pgm](cat1024.pgm) (what is the minimum recurrence time of this image?)
- [cat1368.pgm](cat1368.pgm) (verify that the minimum recurrence time of this image is 36)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include "pgmutils.h"

/**
 * Compute the `k`-th iterate of the cat map for image `img`. The
 * width and height of the image must be equal. This function must
 * replace the bitmap of `img` with the one resulting after ierating
 * `k` times the cat map. To do so, the function allocates a temporary
 * bitmap with the same size of the original one, so that it reads one
 * pixel from the "old" image and copies it to the "new" image. After
 * each iteration of the cat map, the role of the two bitmaps are
 * exchanged.
 */
void cat_map( PGM_image* img, int k )
{
    int i, x, y;
    const int N = img->width;
    unsigned char *cur = img->bmap;
    unsigned char *next = (unsigned char*)malloc( N*N*sizeof(unsigned char) );
    unsigned char *tmp;

    assert(next != NULL);
    assert(img->width == img->height);

    /* [TODO] Which of the following loop(s) can be parallelized? */
    for (i=0; i<k; i++) {
#ifndef SERIAL
        /* Note: the collapse(2) directive automatically makes the
           loop variables y and x private */
#pragma omp parallel for collapse(2) default(none) shared(cur,next,tmp,img,N)
#endif
        for (y=0; y<N; y++) {
            for (x=0; x<N; x++) {
                const int xnext = (2*x+y) % N;
                const int ynext = (x + y) % N;
                next[ynext*N + xnext] = cur[x+y*N];
            }
        }
        /* Swap old and new */
        tmp = cur;
        cur = next;
        next = tmp;
    }
    img->bmap = cur;
    free(next);
}

void cat_map_interchange( PGM_image* img, int k )
{
    int i, x, y;
    const int N = img->width;
    unsigned char *cur = img->bmap;
    unsigned char *next = (unsigned char*)malloc( N*N*sizeof(unsigned char) );

    assert(next != NULL);
    assert(img->width == img->height);

    /* [TODO] Which of the following loop(s) can be parallelized? */
#ifndef SERIAL
#pragma omp parallel for collapse(2) default(none) shared(N,cur,next,k) private(i)
#endif
    for (y=0; y<N; y++) {
        for (x=0; x<N; x++) {
            /* Compute the k-th iterate of pixel (x, y) */
            int xcur = x, ycur = y;
            for (i=0; i<k; i++) {
                const int xnext = (2*xcur+ycur) % N;
                const int ynext = (xcur + ycur) % N;
                xcur = xnext;
                ycur = ynext;
            }
            next[ycur*N+xcur] = cur[y*N+x];
        }
    }
    img->bmap = next;
    free(cur);
}

int main( int argc, char* argv[] )
{
    PGM_image img;
    int niter;
    double elapsed;
    const int NTESTS = 5; /* number of replications */

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s niter < input > output\n", argv[0]);
        return EXIT_FAILURE;
    }
    niter = atoi(argv[1]);
    read_pgm(stdin, &img);

    if ( img.width != img.height ) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }

    /**
     ** WITHOUT loop interchange
     **/
    elapsed = 0.0;
    for (int i=0; i<NTESTS; i++) {
        fprintf(stderr, "Run %d of %d\n", i+1, NTESTS);
        const double tstart = hpc_gettime();
        cat_map(&img, niter);
        elapsed += hpc_gettime() - tstart;
        if (i==0)
            write_pgm(stdout, &img, "produced by omp-cat-map.c");
    }
    elapsed /= NTESTS;

    fprintf(stderr, "\n=== Without loop interchange ===\n");
#if defined(_OPENMP)
    fprintf(stderr, "  OpenMP threads : %d\n", omp_get_max_threads());
#else
    fprintf(stderr, "  OpenMP disabled\n");
#endif
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "        Mops/sec : %f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n\n", elapsed);

    /**
     ** WITH loop interchange
     **/
    elapsed = 0.0;
    for (int i=0; i<NTESTS; i++) {
        fprintf(stderr, "Run %d of %d\n", i+1, NTESTS);
        const double tstart = hpc_gettime();
        cat_map_interchange(&img, niter);
        elapsed += hpc_gettime() - tstart;
    }
    elapsed /= NTESTS;

    fprintf(stderr, "\n=== With loop interchange ===\n");
#if defined(_OPENMP)
    fprintf(stderr, "  OpenMP threads : %d\n", omp_get_max_threads());
#else
    fprintf(stderr, "  OpenMP disabled\n");
#endif
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "        Mops/sec : %f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n\n", elapsed);

    free_pgm( &img );
    return EXIT_SUCCESS;
}
