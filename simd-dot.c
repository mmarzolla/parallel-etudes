/****************************************************************************
 *
 * simd-dot.c - SIMD Dot product
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 ******************************************************************************/

/***
% HPC - SIMD Dot product
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-11-09

## Environment setup

To see which SIMD extensions are supported by the CPU you can examine
the output of `cat /proc/cpuinfo` or `lscpu`. Look at the _flags_
field for the presence of the abbreviations `mmx`,` sse`, `sse2`,`
sse3`, `sse4_1`,` sse4_2`, `avx`,` avx2`.

Compile SIMD programs with:

        gcc -std=c99 -Wall -Wpedantic -O2 -march=native -g -ggdb prog.c -o prog

where:

- `-march=native` enables all statements supported by the
  machine on which you are compiling;

- `-g -ggdb` generates debugging information; this is useful for
  showing the source code along with the corresponding assembly code
  (see below).

It is sometimes useful to analyze the assembly code produced by the
compiler, e.g., to see if SIMD instructions have actually been
emitted. This can be done with the command:

        objdump -dS executable_name

Use the following command to see which compiler flags are enabled by
`-march=native`:

        gcc -march=native -Q --help=target

## Scalar product

[simd-dot.c](simd-dot.c) contains a function that computes the scalar
product of two arrays. The program prints the mean execution times of
the serial and SIMD versions; the goal of this exercise is to develop
the SIMD version. The dot product requires little time even with large
arrays; therefore, you might not observe a significant spèeedup.

**1. Auto-vectorization.** Check the effectiveness of compiler
auto-vectorization of `scalar_dot()`. Compile as follows:

        gcc -O2 -march=native -ftree-vectorize -fopt-info-vec-all \
          simd-dot.c -o simd-dot -lm 2>&1 | grep "loop vectorized"

The `-ftree-vectorize` enables auto-vectorization;
`-fopt-info-vec-all` flag prints some "informative" messages (so to
speak) on standard error to show which loops have been vectorized.

Recent versions of GCC correctly vectorize the `serial_dot()`
function. Older versions vectorize the loop in the `fill()` function,
but not that in `serial_dot()`.

**2. Auto-vectorization (second attempt).** Examine the assembly code
to verify that SIMD instructions have indeed been emitted:

        gcc -S -c -march=native -O2 -ftree-vectorize simd-dot.c -o simd-dot.s

If you have an older version of GCC, examine the diagnostic messages
of the compiler (remove the strings from `2>&1` onwards from the
previous command); you should see something like:

        simd-dot.c:157:5: note: reduction: unsafe fp math optimization: r_17 = _9 + r_20;

that refers to the "for" loop of the `scalar_dot()` function. The
message reports that the instructions:

        r += x[i] * y[i];

are part of a reduction operation involving operands of type
`float`. Since floating-point arithmetic is not commutative, the
compiler did not vectorize in order not to alter the order of the
sums. To ignore the problem, recompile the program with the
`-funsafe-math-optimizations` flag:

        gcc -O2 -march=native -ftree-vectorize -fopt-info-vec-all \
          -funsafe-math-optimizations \
          simd-dot.c -o simd-dot -lm 2>&1 | grep "loop vectorized"

The following message should now appear:

        simd-dot.c:165:5: optimized: loop vectorized using 32 byte vectors

**3. Vectorize the code manually.** Implement `simd_dot()` using the
vector datatypes of the GCC compiler. The function should be very
similar to the one computing the sum-reduction (refer to
`simd-vsum-vector.c` in the examples archive). The function
`simd_dot()` should work correctly for any length $n$ of the input
arrays, which is therefore not required to be a multiple of the SIMD
array lenght. Input arrays are always correctly aligned.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -O2 -march=native -g -ggdb simd-dot.c -o simd-dot -lm

(do _not_ use `-ftree-vectorize`, since we want to compare the
execution time of the pure scalar version with the hand-tuned SIMD
implementation).

Run with:

        ./simd-dot [n]

Example:

        ./simd-dot 20000000

## Files

- [simd-dot.c](simd-dot.c)
- [hpc.h](hpc.h)

 ***/

/* The following #define is required by posix_memalign() and MUST
   appear before including any system header */
#define _XOPEN_SOURCE 600

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <strings.h> /* for bzero() */
#include <math.h> /* for fabs() */

typedef float v4f __attribute__((vector_size(16)));
#define VLEN (sizeof(v4f)/sizeof(float))

/* Returns the dot product of arrays x[] and y[] of legnth n */
float serial_dot(const float *x, const float *y, int n)
{
    double r = 0.0; /* use double here to avoid some nasty rounding errors */
    int i;
    for (i=0; i<n; i++) {
        r += x[i] * y[i];
    }
    return r;
}

/* Same as above, but using the vector datatype of GCC */
float simd_dot(const float *x, const float *y, int n)
{
#ifdef SERIAL
    /* [TODO] */
    return 0;
#else
    v4f vr = {0.0f, 0.0f, 0.0f, 0.0f};
    float r = 0.0f;
    const v4f *vx = (v4f*)x;
    const v4f *vy = (v4f*)y;
    int i;
    for (i=0; i<n-VLEN+1; i += VLEN) {
        vr += (*vx) * (*vy);
        vx++;
        vy++;
    }
    for ( ; i<n; i++) {
        r += x[i] * y[i];
    }
    for (i=0; i<VLEN; i++) {
        r += vr[i];
    }
    return r;
#endif
}

/* Initialize vectors x and y */
void fill(float* x, float* y, int n)
{
    int i;
    const float xx[] = {-2.0f, 0.0f, 4.0f, 2.0f};
    const float yy[] = { 1.0f/2.0, 0.0f, 1.0/16.0, 1.0f/2.0f};
    const size_t N = sizeof(xx)/sizeof(xx[0]);

    for (i=0; i<n; i++) {
        x[i] = xx[i % N];
        y[i] = yy[i % N];
    }
}

int main(int argc, char* argv[])
{
    const float TOL = 1e-5;
    const int nruns = 10; /* number of replications */
    int r, n = 10*1024*1024;
    double serial_elapsed, simd_elapsed;
    double tstart, tend;
    float *x, *y, serial_result, simd_result;
    int ret;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    assert(n > 0);

    const size_t size = n * sizeof(*x);

    assert( size < 1024*1024*200UL );
    ret = posix_memalign((void**)&x, __BIGGEST_ALIGNMENT__, size);
    assert( 0 == ret );
    ret = posix_memalign((void**)&y, __BIGGEST_ALIGNMENT__, size);
    assert( 0 == ret );

    printf("Array length = %d\n", n);

    fill(x, y, n);
    /* Collect execution time of serial version */
    serial_elapsed = 0.0;
    for (r=0; r<nruns; r++) {
        tstart = hpc_gettime();
        serial_result = serial_dot(x, y, n);
        tend = hpc_gettime();
        serial_elapsed += tend - tstart;
    }
    serial_elapsed /= nruns;

    fill(x, y, n);
    /* Collect execution time of the parallel version */
    simd_elapsed = 0.0;
    for (r=0; r<nruns; r++) {
        tstart = hpc_gettime();
        simd_result = simd_dot(x, y, n);
        tend = hpc_gettime();
        simd_elapsed += tend - tstart;
    }
    simd_elapsed /= nruns;

    printf("Serial: result=%f, avg. time=%f (%d runs)\n", serial_result, serial_elapsed, nruns);
    printf("SIMD  : result=%f, avg. time=%f (%d runs)\n", simd_result, simd_elapsed, nruns);

    if ( fabs(serial_result - simd_result) > TOL ) {
        fprintf(stderr, "Check FAILED\n");
        return EXIT_FAILURE;
    }

    printf("Speedup (serial/SIMD) %f\n", serial_elapsed / simd_elapsed);

    free(x);
    free(y);
    return EXIT_SUCCESS;
}
