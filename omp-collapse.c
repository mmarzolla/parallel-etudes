/****************************************************************************
 *
 * omp-collapse.c - Manual implementation of the cllapse() directive
 *
 * Copyright (C) 2026 Moreno Marzolla
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
% Mandelbrot set
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-06-09

Function `test()` contains three nested loops that are collapsed using
the `collapse(3)` OpenMP directive§; a `reduction()` clause is used to
handle a reduction.

The goal of this exercise is to collapse the loops without using the
`collapse()` directive; in other words, it is required to rewrite
the three nested loops:

```C
for (int i=start_i; i < end_i; i += di) {
    for (int j=start_j; j < end_j; j += dj) {
        for (int k=start_k; k < end_k; k += dk) {
            \/\* body \*\/
        }
    }
}
```

as a single loop by introducint a new variable `idx` that needs to
be "unpacked" inside the loop to recover the values of the index
variables `i`, `j`, `k` of the original loops:

```C
for (int idx=...; ...; ...) {
    int i = ...;
    int j = ...;
    int k = ...;
    \/\* body \*\/
}
```

## Files

- [omp-collapse.c](omp-collapse.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <omp.h>

int test( void )
{
    int result = 0;
    const int start_i=0, end_i=3, di=1;
    const int start_j=4, end_j=81, dj=17;
    const int start_k=3, end_k=8, dk=2;
    assert(start_i < end_i);
    assert(di > 0);
    assert(start_j < end_j);
    assert(dj > 0);
    assert(start_k < end_k);
    assert(dj > 0);
#ifdef SERIAL
#pragma omp parallel for collapse(3) reduction(+:result)
    for (int i=start_i; i < end_i; i += di) {
        for (int j=start_j; j < end_j; j += dj) {
            for (int k=start_k; k < end_k; k += dk) {
                //printf("i=%d j=%d k=%d\n", i, j, k);
                result += i + j + k;
            }
        }
    }
    return result;
#else
    const int niter_i = (end_i - start_i - 1) / di + 1;
    const int niter_j = (end_j - start_j - 1) / dj + 1;
    const int niter_k = (end_k - start_k - 1) / dk + 1;
#pragma omp parallel for reduction(+:result)
    for (int idx=0; idx<niter_i * niter_j * niter_k; idx++) {
        int tmp = idx;
        const int k = start_k + (tmp % niter_k)*dk;
        tmp /= niter_k;
        const int j = start_j + (tmp % niter_j)*dj;
        tmp /= niter_j;
        const int i = start_i + tmp * di;
        //printf("i=%d j=%d k=%d\n", i, j, k);
        result += i + j + k;
    }
    return result;
#endif
}

int main( void )
{
    const int result = test();
    printf("%d\n", result);
    return EXIT_SUCCESS;
}
