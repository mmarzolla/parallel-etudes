/****************************************************************************
 *
 * simd-cat-map.c - Arnold's cat map
 *
 * Copyright (C) 2016--2025 Moreno Marzolla
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
% Ardnold's cat map
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-09

![](cat-map.png)

The goal of this exercise is to write a SIMD version of a program to
compute the iterate of _Arnold's cat map_. We have already seen this
problem in other lab sessions; to make this exercise self-contained,
we report here the problem specification.

[Arnold's cat map](https://en.wikipedia.org/wiki/Arnold%27s_cat_map)
is a continuous chaotic function that has been studied in the '60s by
the Russian mathematician [Vladimir Igorevich
Arnold](https://en.wikipedia.org/wiki/Vladimir_Arnold). In its
discrete version, the function can be understood as a transformation
of a bitmap image $P$ of size $N \times N$ into a new image $P'$ of
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

![Figure 1: Arnold's cat map.](cat-map.svg)

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
iterations that depends on $N$ and has been proved to never exceed
$3N$, we get back the original image! (Figure 2).

![Figure 2: Some iterations of the cat map.](cat-map-demo.png)

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

You are given sequential program that computes the $k$-th iterate of
the cat map using the CPU. The number of iterations $k$ to compute is
passed on the command line. The program reads an image in PGM format
from standard input, and produces to standard output the image that is
produced after $k$ iterations of the cat map. You should redirect the
standard output to a file, as shown in the comments to the source
code.

The structure of the function that calculates the $k$-th iterate of
the cat map is very simple:

```C
for (int y=0; y<N; y++) {
	for (int x=0; x<N; x++) {
		\/\* compute the coordinates (xnew, ynew) of point (x, y)
                     after k iterations of the cat map \*\/
		next[xnew + ynew*N] = cur[x+y*N];
	}
}
```

To make use of SIMD parallelism we proceed as follows: instead of
computing the new coordinates of a single point at a time, we compute
the new coordinates of all adjacent points $(x, y)$, $(x+1,y)$,
$(x+2,y)$, $(x+3,y)$ using the compiler's _vector datatype_. To this
aim, we define the following variables of type `v4i` (i.e., SIMD array
of four integers):

- `vx`, `vy`: $x$ and $y$ coordinates of four adjacent points, before
  the application of the cat map; the coordinates of point $i=0,
  \ldots 3$ are `(vx[i], vy[i])`.

- `vxnew`, `vynew`: new coordinates of four adjacent points, after
  application of the cat map.

Type `v4i` is defined as:

```C
	typedef int v4i __attribute__((vector_size(16)));
	#define VLEN (sizeof(v4i)/sizeof(int))
```

Let $vx = \{x, x+1, x+2, x+3\}$, $vy = \{y, y, y, y\}$; we can apply
the cat map to all $vx$, $vy$ using the usual C operators, i.e., you
should not need to change the instructions that actually compute the
new coordinates.

There is, however, one exception. The following scalar instruction:

```C
	next[xnew + ynew*N] = cur[x+y*N];
```

can not be parallelized automatically by the compiler if `x`, `y`,
`xnew`, `ynew` are SIMD vectors. Instead, you must expand the
instruction into the following four lines of code:

```C
	next[vxnew[0] + vynew[0]*N] = cur[vx[0] + vy[0]*N];
	next[vxnew[1] + vynew[1]*N] = cur[vx[1] + vy[1]*N];
	next[vxnew[2] + vynew[2]*N] = cur[vx[2] + vy[2]*N];
	next[vxnew[3] + vynew[3]*N] = cur[vx[3] + vy[3]*n];
```

You can assume that the size $N$ of the image is always
an integer multiple of the SIMD vector length, i.e., an integer
multiple of 4.

To compile:

        gcc -std=c99 -Wall -Wpedantic -O2 -march=native simd-cat-map.c -o simd-cat-map

To execute:

        ./simd-cat-map k < input_file > output_file

Example:

        ./simd-cat-map 100 < cat368.pgm > cat1368-100.pgm

## Extension

The performance of the SIMD version should be only marginally better
than the version scale (actually, the SIMD version could even be
_worse_ than the serial version). Analyzing the assembly code produced
by the compiler, it turns out that the computation of the modulus
operator in the expressions

```C
	vxnew = (2*vxold+vyold) % N;
	vynew = (vxold + vyold) % N;
```

is done using scalar operations. By consulting the list of _SIMD
intrinsics_ on [Intel's web
site](https://software.intel.com/sites/landingpage/IntrinsicsGuide/),
we see that there is no SIMD instruction for integer division.
Therefore, to get better performance it is essential to compute the
modulus without using division at all.

Analyzing the scalar code, we realize that if $0 \leq xold < N$ and $0
\leq yold < N$, then we have $0 \leq 2 \times xold + yold < 3N$ and $0
\leq xold+yold < 2N$. Therefore, the scalar code can be rewritten as:

```C
	xnew = (2*xold + yold);
	if (xnew >= N) { xnew = xnew - N; }
	if (xnew >= N) { xnew = xnew - N; }
	ynew = (xold + yold);
	if (ynew >= N) { ynew = ynew - N; }
```

The code above is certainly more verbose and less readable than the
original version that uses the modulus operator, but it has the
advantage of not requiring the modulus operator nor integer division.
Furthermore, we can use the "selection and masking" technique to
parallelize the conditional statements. Indeed, the instruction

```C
	if (xnew >= N) { xnew = xnew - N; }
```

can be rewritten as

```C
	const v4i mask = (xnew >= N);
	xnew = (mask & (xnew - N)) | (mask & xnew);
```

that tan be further simplified as:

```C
	const v4i mask = (xnew >= N);
	xnew = xnew - (mask & N);
```

The SIMD program becomes more complex, but at the same time more
efficient than the serial program.

To compile:

        gcc -std=c99 -Wall -Wpedantic -march=native -O2 simd-cat-map.c -o simd-cat-map

To execute:

        ./simd-cat-map [niter] < in.pgm > out.pgm

Example:

        ./simd-cat-map 1024 < cat1368.pgm > cat1368-1024.pgm

## Files

- [simd-cat-map.c](simd-cat-map.c)
- [hpc.h](hpc.h)
- [cat1368.pgm](cat1368.pgm) (the minimum recurrence time of this image is 36)

 ***/

/* The following #define is required by posix_memalign() and
   clock_gettime(). It MUST be defined before including any other
   file. */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"

typedef int v4i __attribute__((vector_size(16)));
#define VLEN (sizeof(v4i)/sizeof(int))

#include "pgmutils.h"

/**
 * Compute the |k|-th iterate of the cat map for image |img|. You must
 * implement this function, starting with a serial version, and then
 * adding OpenMP pragmas.
 */
void cat_map( PGM_image* img, int k )
{
    const int N = img->width;
    unsigned char *cur = img->bmap;
    unsigned char *next;
    int ret;

    assert( img->width == img->height );
    assert( img->width % VLEN == 0);

    ret = posix_memalign((void**)&next, __BIGGEST_ALIGNMENT__, N*N*sizeof(*next));
    assert( 0 == ret );

#ifdef SERIAL
    for (int y=0; y<N; y++) {
        /* [TODO] The SIMD version should compute the new position of
           four adjacent pixels (x,y), (x+1,y), (x+2,y), (x+3,y) using
           SIMD instructions. Assume that w (the image width) is
           always a multiple of VLEN. */
        for (int x=0; x<N; x++) {
            int xold = x, xnew = xold;
            int yold = y, ynew = yold;
            for (int i=0; i<k; i++) {
                xnew = (2*xold+yold) % N;
                ynew = (xold + yold) % N;
                xold = xnew;
                yold = ynew;
            }
            next[xnew + ynew*N] = cur[x+y*N];
        }
    }
#else
    /* SIMD version of the cat map iteration. The idea is to compute
       the new coordinates of four adjacent points (x, y), (x+1,y),
       (x+2,y) and (x+3,y) using SIMD instructions. To do so, we pack
       the coordinates into two SIMD registers vx = {x, x+1, x+2,
       x+3}, vy = {y, y, y, y}.

       Note that initially vx = {0, 1, 2, 3}, and at each iteration of
       the inner loop, all elements of vx are incremented by
       VLEN. Therefore, assuming VLEN == 4, at the second iteration we
       have vx = {4, 5, 6, 7}, then {8, 9, 10, 11} and so on. */
    for (int y=0; y<N; y++) {
        v4i vx = {0, 1, 2, 3};
        const v4i vy = {y, y, y, y};
        for (int x=0; x<N-VLEN+1; x += VLEN) {
            v4i xold = vx, xnew = xold;
            v4i yold = vy, ynew = yold;
            for (int i=0; i<k; i++) {
#if 0
                xnew = (2*xold+yold) % N;
                ynew = (xold + yold) % N;
#else
                /* There is no SIMD instruction for integer division
                   in SSEx/AVX (_mm_div_epi32() exists on Intel MIC
                   only), so computing the remainder requires scalar
                   division and is extremely slow, so,
                   auto-vectorization does not really work in this
                   case.

                   The code below gets rid of the integer division, at
                   the cost of additional code complexity.  On my
                   Intel i7-4790 processor with gcc 7.5.0 the code
                   below is 10x faster than using the modulo operator
                   with auto-vectorization.

                   The situation is similar on ARM: on ARMv7 (armv7l,
                   Raspberry Pi4, gcc 8.3.0) there is no SIMD integer
                   division and the code below requires 66% the time
                   of the one using the modulo operator.
                */
                v4i mask;
                /* assuming 0 <= xold < N and 0 <= yold < N, we have
                   that 0 <= xnew < 3N; therefore, we might need to
                   subtract N at most twice from xnew to compute the
                   correct remainder modulo N. */
                xnew = (2*xold+yold);

                mask = (xnew >= N);
                xnew = (mask & (xnew - N)) | (~mask & xnew); // or: xnew = xnew - (mask & N);
                mask = (xnew >= N);
                xnew = (mask & (xnew - N)) | (~mask & xnew); // or: xnew = xnew - (mask & N);

                /* assuming 0 <= xold < N and 0 <= yold < N, we have
                   that 0 <= ynew < 2N; therefore, we might need to
                   subtract N at most once. */
                ynew = (xold + yold);
                mask = (ynew >= N);
                ynew = (mask & (ynew - N)) | (~mask & ynew); // or: ynew = ynew - (mask & N);
#endif
                xold = xnew;
                yold = ynew;
            }
            next[xnew[0] + ynew[0]*N] = cur[vx[0]+y*N];
            next[xnew[1] + ynew[1]*N] = cur[vx[1]+y*N];
            next[xnew[2] + ynew[2]*N] = cur[vx[2]+y*N];
            next[xnew[3] + ynew[3]*N] = cur[vx[3]+y*N];
            vx += VLEN;
        }
    }
#endif

    img->bmap = next;
    free(cur);
}

int main( int argc, char* argv[] )
{
    PGM_image img;
    int niter;

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s niter < in.pgm > out.pgm\n\nExample: %s 684 < cat1368.pgm > out1368.pgm\n", argv[0], argv[0]);
        return EXIT_FAILURE;
    }
    niter = atoi(argv[1]);
    read_pgm(stdin, &img);
    if ( img.width != img.height ) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }
    if ( img.width % VLEN ) {
        fprintf(stderr, "FATAL: this program expects the image width (%d) to be a multiple of %d\n", img.width, (int)VLEN);
        return EXIT_FAILURE;
    }
    const double tstart = hpc_gettime();
    cat_map(&img, niter);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "    SIMD width: %d bytes\n", (int)VLEN);
    fprintf(stderr, "    Iterations: %d\n", niter);
    fprintf(stderr, "  Width,Height: %d,%d\n", img.width, img.height);
    fprintf(stderr, "      Mops/sec: %.4f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Execution time  %.3f\n", elapsed);

    write_pgm(stdout, &img, "produced by simd-cat-map.c");
    free_pgm(&img);
    return EXIT_SUCCESS;
}
