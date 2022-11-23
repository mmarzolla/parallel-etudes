/****************************************************************************
 *
 * omp-cat-map.c - Arnold's cat map
 *
 * Copyright (C) 2016--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% Last updated: 2022-10-18

![Vladimir Igorevich Arnold (1937--2010). By Svetlana Tretyakova - <http://www.mccme.ru/arnold/pool/original/VI_Arnold-05.jpg>, CC BY-SA 3.0](Vladimir_Arnold.jpg)

[Arnold's cat map](https://en.wikipedia.org/wiki/Arnold%27s_cat_map)
is a continuous chaotic function that has been studied in the '60s by
the Russian mathematician [Vladimir Igorevich
Arnold](https://en.wikipedia.org/wiki/Vladimir_Arnold). In its
discrete version, the function can be understood as a transformation
of a bitmapped image $P$ of size $N \times N$ into a new image $P'$ of
the same size. For each $0 \leq x, y < N$, the pixel of coordinates
$(x,y)$ in $P$ is mapped into a new position $C(x, y) = (x', y')$ in
$P'$ where

$$
x' = (2x + y) \bmod N, \qquad y' = (x + y) \bmod N
$$

("mod" is the integer remainder operator, i.e., operator `%` of the C
language). We may assume that $(0, 0)$ is top left and $(N-1, N-1)$
bottom right, so that the bitmap can be encoded as a regular
two-dimensional C matrix.

The transformation corresponds to a linear "stretching" of the image,
that is then broken down into triangles that are rearranged as shown
in Figure 1.

![Figura 1: Arnold's cat map](cat-map.png)

Arnold's cat map has interesting properties. Let $C^k(x, y)$ be the
result of iterating $k$ times the function $C$, i.e.:

$$
C^k(x, y) = \begin{cases}
(x, y) & \mbox{if $k=0$}\\
C(C^{k-1}(x,y)) & \mbox{if $k>0$}
\end{cases}
$$

Therefore, $C^2(x,y) = C(C(x,y))$, $C^3(x,y) = C(C(C(x,y)))$, and so
on.

If we take an image and apply $C$ once, we get a severely distorted
version of the input. If we apply $C$ on the resulting image, we get
an even more distorted image. As we keep applying $C$, the original
image is no longer discernible. However, after a certain number of
iterations that depend on $N$ and has been proved to never exceed
$3N$, we get back the original image! (Figure 2).

![Figura 2: Some iterations of the cat map](cat-map-demo.png)

The _minimum recurrence time_ for an image is the minimum positive
integer $k \geq 1$ such that $C^k(x, y) = (x, y)$ for all $(x, y)$. In
simple terms, the minimum recurrence time is the minimum number of
iterations of the cat map that produce the starting image.

For example, the minimum recurrence time for
[cat1368.pgm](cat1368.pgm) of size $1368 \times 1368$ is $36$. As said
before, the minimum recurrence time depends on the image size $N$.
Unfortunately, no closed formula is known to compute the minimum
recurrence time as a function of $N$, although there are results and
bounds that apply to specific cases.

You are provided with a serial program that computes the $k$-th
iterate of Arnold's cat map on a square image. The program reads the
input from standard input in
[PGM](https://en.wikipedia.org/wiki/Netpbm) (_Portable GrayMap_)
format. The results is printed to standard output in PGM format. For
example:

        ./omp-cat-map 100 < cat1368.pgm > cat1368-100.pgm

applies the cat map $k=100$ times on `cat1368.phm` and saves the
result to `cat1368-100.pgm`.

To display a PGM image you might need to convert it to a different
format, e.g., JPEG. Under Linux you can use `convert` from the
[ImageMagick](https://imagemagick.org/) package:

        convert cat1368-100.pgm cat1368-100.jpeg

Modify the function `cat_map()` to make use of shared-memory
parallelism using OpenMP. To this aim it is important to know that
Arnold's cat map is invertible: this means that any two different
points $(x_1, y_1)$ and $(x_2, y_2)$ are always mapped to different
points $(x'_1, y'_1) = C(x_1, y_1)$ and $(x'_2, y'_2) = C(x_2, y_2)$,
suggesting that the destination bitmap $P'$ can be filled concurrently
without race conditions (however, see below for some caveats).

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-cat-map.c -o omp-cat-map

To execute:

        ./omp-cat-map k < input_file > output_file

Example:

        ./omp-cat-map 100 < cat1368.pgm > cat1368-100.pgm

## Suggestions

The provided implementation of function `cat_map()` is based on the
following template:

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

The two innermost loops build a $P'$ from $P$; the outermost loop
applies this transformation $k$ times, using the result of the
previous iteration as the source image. Therefore, the outermost loop
can _not_ be parallelized due to a loop-carried dependence (the result
of an iteration is used as input for the next iteration).

Therefore, in the version above you can either:

1. Parallelize the `y` loop only, or

2. Parallelize the `x` loop only, or

3. Parallelize both the `y` and `x` loops using the `collapse(2)`
   clause.

(I suggest to try option 3. Option 2, although formally correct, does
not appear to be efficient in practice: why?).

We can apply the _loop interchange_ transformation to rewrite the
code above as follows:

```C
for (y=0; y<N; y++) {
        for (x=0; x<N; x++) {
                xcur= x; ycur = y;
                for (i=0; i<k; i++) {
                        (xnext, ynext) = C(xcur, ycur);
                        xcur = xnext;
                        ycur = ynext;
                }
                P'(xnext, ynext) = P(x, y)
        }
}
```

This version can be understood as follows: the two outermost loops
iterate over all pixels $(x, y)$. For each pixel, the innermost loop
computes the target position $(\mathit{xnext}, \mathit{ynext}) =
C^k(x,y)$ that the pixel of coordinates $(x, y)$ will occupy after $k$
iterations of the cat map.

In this second version, we have the following options:

a. Parallelize the outermost loop on `y`, or

b. Parallelize the middle loop on `x`, or

c. Parallelize the two outermost loops with the `collapse(2)` directive.

(I suggest to try option c).

Which is more efficient in your system: option 3 above, or option c?

## To probe further

What is the minimum recurrence time of image
[cat1024.pgm](cat1024.pgm) of size $1024 \times 1024$? Since there is
no general formula, to answer this question we need to iterate the cat
map and stop as soon as we get an image that is equal to the original
one.

It turns out that there is a smarter way, that does not involve slow
comparisons of images. There is actually no need to have an input
image at all: only its size $N$ is required.

To see how it works, let us suppose that we know that one particular
pixel of the image, say $(x_1, y_1)$, has minimum recurrence time
equal to 15. This means that after 15 iterations of the cat map, the
pixel at coordinates $(x_1, y_1)$ will return to its starting
position.

Suppose that another pixel of coordinates $(x_2, y_2)$ has minimum
recurrence time 21. How many iterations of the cat map are required to
have _both_ pixels back to their original positions?

The answer is $105$, which is the least common multiple (LCM) of 15
and 21. From this observation we can devise the following algorithmn
for computing the minimum recurrence time of an image of size $N
\times N$. Let $T(x,y)$ be the minimum recurrence time of the pixel of
coordinates $(x, y)$, $0 \leq x, y < N$. Then, the minimum recurrence
time of the whole image is the least common multiple of all $T(x, y)$.

[omp-cat-map-rectime.c](omp-cat-map-rectime.c) contains an incomplete
skeleton of a program that computes the minimum recurrence time of a
square image of size $N \times N$. A function that computes the LCM of
two positive integers is provided therein. Complete the program and
then produce a parallel version using the appropriate OpenMP
directives.

Table 1 shows the minimum recurrence time for some $N$.

:Tabella 1: Minimum recurrence time for some image sizes $N$

    $N$   Minimum recurrence time
------- -------------------------
     64                        48
    128                        96
    256                       192
    512                       384
   1368                        36
------- -------------------------

Figure 3 shows the minimum recurrence time as a function of
$N$. Despite the fact that the values "jump" from size to size, we see
that they tend to align along straight lines.

![Figura 3: Minimum recurrence time as a function of the image size $N$](cat-map-rectime.png)

## Files

- [omp-cat-map.c](omp-cat-map.c)
- [omp-cat-map-rectime.c](omp-cat-map-rectime.c)
- [cat1024.pgm](cat1024.pgm) (what is the minimum recurrence time of this image?)
- [cat1368.pgm](cat1368.pgm) (verify that the minimum recurrence time of this image is 36)

***/
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

#ifdef SERIAL
    /* [TODO] Which of the following loop(s) can be parallelized? */
#endif
    for (i=0; i<k; i++) {
#ifndef SERIAL
#if __GNUC__ < 9
#pragma omp parallel for collapse(2) default(none) shared(cur,next,tmp,img)
#else
#pragma omp parallel for collapse(2) default(none) shared(cur,next,tmp,img,N)
#endif
#endif
        for (y=0; y<N; y++) {
            for (x=0; x<N; x++) {
                const int xnext = (2*x+y) % N;
                const int ynext = (x + y) % N;
                next[xnext + ynext*N] = cur[x+y*N];
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

#ifndef SERIAL
/**
 * We can interchange the loops to move the parallelization to the
 * outer loop, as explained in the text. To analyze the result, I have
 * executed the program on the systems below. The program has been
 * compiled by the Makefile (OpenMP enabled, default compiler
 * settings) and the command line was:
 *
 * ./omp-cat-map 684 < cat1368.pgm > /dev/null
 *
 * using all logical cores (default behavior of OpenMP). Running times
 * were computed by averaging 5 runs. "N+N cores" means that the
 * processor has N physical cores with hyperthreading, that are seen
 * by the OS as N+N "virtual" cores.
 *
 *                                cat_map()   cat_map_interchange()
 * ----------------------------------------------------------------
 * Intel Xeon E3-1220 (4 cores)      2.64               4.44
 * Intel i7-5820K (6+6 cores)        4.60               1.71
 * Intel i7-2600 (4+4 cores)         2.77               2.46
 * Intel i7-4790 (4+4 cores)         2.25               2.05
 * Intel Atom N570 (2+2 cores)      48.25              34.42
 * Raspberry PI 3 (4 cores)         36.69              27.07
 * Intel Xeon E5-2603 (12 cores)     1.79               2.34
 *
 * cat_map_interchange() is faster on all platforms with the notable
 * exception of the Xeon processors where it was almost two times
 * _slower_ than cat_map(). However, there are differences depending
 * on the number of OpenMP threads (e.g., for different values of P
 * one version might be faster or slower than the other).
 */
void cat_map_interchange( PGM_image* img, int k )
{
    int i, x, y;
    const int N = img->width;
    unsigned char *cur = img->bmap;
    unsigned char *next = (unsigned char*)malloc( N*N*sizeof(unsigned char) );

    assert(next != NULL);
    assert(img->width == img->height);

#if __GNUC__ < 9
#pragma omp parallel for collapse(2) default(none) shared(cur,next,k) private(i)
#else
#pragma omp parallel for collapse(2) default(none) shared(N,cur,next,k) private(i)
#endif
    for (y=0; y<N; y++) {
        for (x=0; x<N; x++) {
            /* Compute the k-th iterate of pixel (x, y) */
            int xcur = x, ycur = y, xnext = x, ynext = x;
            for (i=0; i<k; i++) {
                xnext = (2*xcur+ycur) % N;
                ynext = (xcur + ycur) % N;
                xcur = xnext;
                ycur = ynext;
            }
            next[ynext*N+xnext] = cur[y*N+x];
        }
    }
    img->bmap = next;
    free(cur);
}
#endif

int main( int argc, char* argv[] )
{
    PGM_image img;
    int niter;
    double tstart, elapsed;

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s niter\n", argv[0]);
        return EXIT_FAILURE;
    }
    niter = atoi(argv[1]);
    read_pgm(stdin, &img);

    if ( img.width != img.height ) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }

    tstart = omp_get_wtime();
    cat_map(&img, niter);
    elapsed = omp_get_wtime() - tstart;
    fprintf(stderr, "\n=== Without loop interchange ===\n");
    fprintf(stderr, "  OpenMP threads : %d\n", omp_get_max_threads());
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "     Mpixels/sec : %f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);
    write_pgm(stdout, &img, "produced by omp-cat-map.c");

#ifndef SERIAL
    /**
     ** Second version (with loop interchange)
     **/
    tstart = omp_get_wtime();
    cat_map_interchange(&img, niter);
    elapsed = omp_get_wtime() - tstart;
    fprintf(stderr, "\n=== With loop interchange ===\n");
    fprintf(stderr, "  OpenMP threads : %d\n", omp_get_max_threads());
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "     Mpixels/sec : %f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);
#endif
    free_pgm( &img );
    return EXIT_SUCCESS;
}
