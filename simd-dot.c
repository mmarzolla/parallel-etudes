/****************************************************************************
 *
 * simd-dot,c - Dot product
 *
 * Copyright (C) 2017--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Dot product
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-08-24

## Environment setup

Check which SIMD extensions are supported by the CPU by examining the
output of the `cat /proc/cpuinfo` command or the command `lscpu`. Look
in the _flags_ field for the presence of the abbreviations `mmx`,`
sse`, `sse2`,` sse3`, `sse4_1`,` sse4_2`, `avx`,` avx2`.

Compile SIMD programs with:

        gcc -std=c99 -Wall -Wpedantic -O2 -march=native -g -ggdb prog.c -o prog

where:

- `-march=native` enables all statements supported by the
  machine on which you are compiling;

- `-g -ggdb` generates debugging information; this is useful for
  showing the source code along with the corresponding assembly code
  (see below).

It night be useful to analyze the assembly code produced by the
compiler, e.g., to see if SIMD instructions have actually been
emitted. This can be done with the command:

        objdump -dS executable_name

To know which compiler flags are enabled when `-march=native` is
passed to GCC, use the following command:

        gcc -march=native -Q --help=target

## Scalar product

[simd-dot.c](simd-dot.c) contains a function that computes the scalar
product of two `float` arrays.

The program prints the average execution times of the parallel and
serial versions (the goal of this exercise is to develop the SIMD
version). The dot product is a very simple computation that completes
in a very short time even with large arrays. Therefore, you might not
observe significant differences between the performance of the serial
and SIMD versions.

The goal of this exercise is to employ SIMD parallelism in function
`simd_dot()`, according with the following steps:

**1. Auto-vectorization.** Check the effectiveness of compiler
auto-vectorization of the `scalar_dot()` function. Compile the program
as follows:

        gcc -O2 -march=native -ftree-vectorize -fopt-info-vec \
          -o simd-dot -lm 2>&1 | grep "loop vectorized"

The `-fopt-info-vec-XXX` flags print some "informative" messages (so
to speak) on standard error indicating which loops have been
vectorized, if any. The command line above redirects standard error to
standard output, and searches for the string _loop vectorized_ which
should be printed by the compiler when it succesfully vectorizes a
loop.

One such message should be printed: the compiler vectorizes the loop
in function `fill()`, but not the one in function `serial_dot()`.

**2. Auto-vectorization (second attempt).** Examine the diagnostic
messages of the compiler (remove the strings from `2>&1` onwards
from the previous command). The current version of GCC installed
on the server (9.4.0) gives a cryptic message:

        simd-dot.c:165:5: missed: couldn't vectorize loop
        simd-dot.c:166:11: missed: not vectorized: relevant stmt not supported: r_17 = _9 + r_20;

Older versions of GCC gave a more meaningful message:

        simd-dot.c: 165: 5: note: reduction: unsafe fp math optimization: r_17 = _9 + r_20;

Both refer to the "for" loop of the `scalar_dot()` function. The
latter message reports that the instructions:

        r += x[i] * y[i];

are part of a reduction operation involving operands of type
`float`. Since floating-point arithmetic does not enjoy the
commutative property, the compiler does not vectorize in order not to
alter the order of the sums. To ignore the problem, recompile the
program with the `-funsafe-math-optimizations` flag:

        gcc -O2 -march=native -ftree-vectorize -fopt-info-vec \
          -funsafe-math-optimizations \
          simd-dot.c -o simd-dot -lm 2>&1 | grep "loop vectorized"

The following message should now appear:

        simd-dot.c:165:5: optimized: loop vectorized using 32 byte vectors

indicating that the cycle has been vectorized.

**3. Vectorize the code manually.** Implement the function
`simd_dot()` using the _vector datatype_ of the GCC compiler. The
function is very similar to the function for computring the
sum-reduction of an array that we have seen in the class (refer to
`simd-vsum-vector.c` in the examples archive). Function `simd_dot()`
should work correctly for any length $n$ of the input arrays, which is
not required to be a multiple of the SIMD array widths. Input arrays
are always correctly aligned.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -O2 -march=native -g -ggdb simd-dot.c -o simd-dot -lm

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

    if ( fabs(serial_result - simd_result) > 1e-5 ) {
        fprintf(stderr, "Check FAILED\n");
        return EXIT_FAILURE;
    }

    printf("Speedup (serial/SIMD) %f\n", serial_elapsed / simd_elapsed);

    free(x);
    free(y);
    return EXIT_SUCCESS;
}
