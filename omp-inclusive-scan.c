/****************************************************************************
 *
 * omp-inclusive-scan.c - Inclusive scan with OpenMP
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Inclusive Scan
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2022-10-12

The file [omp-inclusive-scan.c](omp-inclusive-scan.c) contains a
serial program that computes the _prefix sum_ of an array `v[]` of
length $n$ by applying the inclusive scan operator; the result is
stored in `s[]`. At the end of the operation we therefore have $s[i] =
v[0] + v[1] + \ldots + v[i]$ for each $i = 0, \ldots, n-1$.

Since in shared-memory programming it is often assumed that the number
$P$ of threads is much lower than the problem size ($P \ll n$), it is
not possible to impelment the tree-based computation shown during the
class, which is more appropriate for GPUs. Therefore, we use a
different strategy, called _blocked scan_ (refer to Figure 1, where $P
= 4$ OpenMP threads are assumed).

![Figure 1: Blocked inclusive scan](omp-inclusive-scan.svg)

Each thread operates on portions of `v[]` and `s[]` whose endpoints are
determined appropriately.

1. Each thread performs an inclusive scan of its own portion of `v[]`
   using the serial algorithm. The result is stored in the
   corresponding portion of `s[]`.

2. The value of the last element of each block of `s[]` is copied to a
   temporary shared array `blksum[]` of length $P$.

3. The master computes the inclusive scan `blksum[]`, storing the
   result on a new array `blksum_s[]`; it would also be possible to
   store the result inside `blksum []` without use a separate array.

4. Each thread $p$ adds `blksum_[p]` to all elements of their own
   portion of `s[]`.

At the end, `s[]` contains the sum-inclusive scan `v[]`

Stages 1 and 4 can be done in parallel by OpenMP threads since they
are embarrassingly parallel.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-inclusive-scan.c -o omp-inclusive-scan

Run with:

        OMP_NUM_THREADS=2 ./omp-inclusive-scan

## Files

- [omp-inclusive-scan.c](omp-inclusive-scan.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

/* Fill v[] with the constant 1 */
void fill(int* v, int n)
{
    int i;
    printf("Initializing %d elements...\n", n);
    for (i=0; i<n; i++) {
        v[i] = 1;
    }
}

void check(int *s, int n)
{
    int i;
    for (i=0; i<n; i++) {
        if ( s[i] != i+1 ) {
            printf("Check failed: expected s[%d]==%d, got %d\n", i, i+1, s[i]);
            abort();
        }
    }
    printf("Check ok!\n");
}

/* Compute the inclusive scan of the n-elements array v[], and store
   the result in s[]. The caller is responsible for allocating s with
   n elements */
void inclusive_scan(int *v, int n, int *s)
{
    int i;
    printf("Scanning %d elements...\n", n);
#ifdef SERIAL
    /* degenerate case of empty array: do nothing */
    if ( n == 0 )
        return;

    /* [TODO] parallelize this */
    s[0] = v[0];
    for (i=1; i<n; i++) {
        s[i] = s[i-1] + v[i];
    }
#else
    const int n_threads = omp_get_max_threads();
    int blksum[n_threads];
    int blksum_s[n_threads];

#pragma omp parallel num_threads(n_threads) default(none) shared(v,s,n,blksum,n_threads) private(i)
    {
        const int thread_id = omp_get_thread_num();
        const int local_start = n * thread_id / n_threads;
        const int local_end = n * (thread_id + 1) / n_threads;

        /* Each process performs an inclusive scan of its portion of array */
        s[local_start] = v[local_start];
        for (i=local_start+1; i<local_end; i++) {
            s[i] = s[i-1] + v[i];
        }
        blksum[thread_id] = s[local_end-1];
    }

    /* The master performs an exclusive scan on blksum_s[] */
    blksum_s[0] = 0;
    for (i=1; i<n_threads; i++) {
        blksum_s[i] = blksum_s[i-1] + blksum[i-1];
    }

    /* Each process increments all values of its portion of array */
#pragma omp parallel num_threads(n_threads) default(none) shared(s,n,blksum_s,n_threads) private(i)
    {
        const int thread_id = omp_get_thread_num();
        const int local_start = n * thread_id / n_threads;
        const int local_end = n * (thread_id + 1) / n_threads;

        for (i=local_start; i<local_end; i++) {
            s[i] += blksum_s[thread_id];
        }
    }
#endif
}

int main( int argc, char *argv[] )
{
    int n = 1000000;
    int *v, *s;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc == 2 ) {
        n = atoi(argv[1]);
    }

    if (n > (1<<29)) {
        fprintf(stderr, "FATAL: array too large\n");
        return EXIT_FAILURE;
    }

    v = (int*)malloc(n*sizeof(int)); assert(v != NULL);
    s = (int*)malloc(n*sizeof(int)); assert(s != NULL);
    fill(v, n);
    inclusive_scan(v, n, s);
    check(s, n);
    free(v);
    free(s);
    return EXIT_SUCCESS;
}
