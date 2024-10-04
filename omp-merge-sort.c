/****************************************************************************
 *
 * omp-merge-sort.c - Merge Sort with OpenMP tasks
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Merge Sort with OpenMP tasks
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2023-10-24

The file [omp-merge-sort.c](omp-merge-sort.c) contains a recursive
implementation of the _Merge Sort_ algorithm. The implementation uses
_Selection Sort_ when the size of the subvector is less than a
user-defined cutoff value; this is a standard optimization that avoids
the overhead of recursive calls on small vectors.

The program generates and sorts a random permutation of $0, 1, \ldots,
n-1$; it if therefore easy to check the correctness of the result,
since it must be the sequente $0, 1, \ldots, n-1$.

The goal is to parallelize the Merge Sort algorithm using OpenMP
tasks. You might want to proceed as follows:

- The recursion must start inside a parallel region;

- Only one process should start the recursion;

- Create two tasks for the two recursive calls; pay attention to the
  visibility of variables;

- Wait for the two sub-tasks to complete before starting the "merge"
  step.

Measure the execution time of the parallel version and compare
it with the serial implementation. To get meaningful results, choose
an input that requires at least a few seconds to be sorted using all
available processor cores.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-merge-sort.c -o omp-merge-sort

To execute:

        ./omp-merge-sort 50000

## Files

- [omp-merge-sort.c](omp-merge-sort.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

int min(int a, int b)
{
    return (a < b ? a : b);
}

void swap(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/**
 * Sort v[low..high] using selection sort. This function will be used
 * for small vectors only. Do not parallelize this.
 */
void selectionsort(int* v, int low, int high)
{
    int i, j;
    for (i=low; i<high; i++) {
        for (j=i+1; j<=high; j++) {
            if (v[i] > v[j]) {
                swap(&v[i], &v[j]);
            }
        }
    }
}

/**
 * Merge src[low..mid] with src[mid+1..high], put the result in
 * dst[low..high].
 *
 * Do not parallelize this function (it could be done, but is very
 * difficult, see
 * http://www.drdobbs.com/parallel/parallel-merge/229204454
 * https://en.wikipedia.org/wiki/Merge_algorithm#Parallel_merge )
 */
void merge(int* src, int low, int mid, int high, int* dst)
{
    int i=low, j=mid+1, k=low;
    while (i<=mid && j<=high) {
        if (src[i] <= src[j]) {
            dst[k] = src[i++];
        } else {
            dst[k] = src[j++];
        }
        k++;
    }
    /* Handle leftovers */
    while (i<=mid) {
        dst[k] = src[i++];
        k++;
    }
    while (j<=high) {
        dst[k] = src[j++];
        k++;
    }
}

/**
 * Sort `v[i..j]` using the recursive version of Merge Sort; the array
 * `tmp[i..j]` is used as a temporary buffer; the caller is
 * responsible for providing a suitably sized array `tmp`. This
 * function must not free `tmp`.
 */
void mergesort_rec(int* v, int i, int j, int* tmp)
{
    const int CUTOFF = 64;
    /* If the subvector is smaller than CUTOFF, use selectoin
       sort. This is a widely used optimization that avoids the
       overhead of recursion for small vectors. The optimal CUTOFF
       value is implementation-dependent; the value used here is just
       an example. */
    if ( j - i + 1 < CUTOFF )
        selectionsort(v, i, j);
    else {
        const int m = (i+j)/2;
        /* [TODO] The two recursive invocations of `mergesort_rec()`
           are independent and can be executed in parallel. Create two
           OpenMP tasks, and wait for their completion before merging
           the results. Pay attention to the visibility of variables
           associated to the tasks.

           `v`, `i`, `m`, `tmp` are local variables, so they are
           `firstprivate` by default according to the visibility rules
           for tasks. However, due to the `taskwait` directive below,
           the values of these variables can not change between task
           creation and execution, so they can be made all
           `shared`. */
#ifndef SERIAL
#pragma omp task shared(v, i, m, tmp)
#endif
        mergesort_rec(v, i, m, tmp);
#ifndef SERIAL
#pragma omp task shared(v, j, m, tmp)
#endif
        mergesort_rec(v, m+1, j, tmp);
        /* Wait for completion of the recursive invocations of
           `mergesort_rec()` before merging. */
#ifndef SERIAL
#pragma omp taskwait
#endif
        merge(v, i, m, j, tmp);
        /* copy the sorted data back to v */
        memcpy(v+i, tmp+i, (j-i+1)*sizeof(v[0]));
    }
}

/**
 * Sort v[] of length n using Merge Sort; after allocating a temporary
 * array with the same size of a (used for merging), this function
 * just calls mergesort_rec with the appropriate parameters.  After
 * mergesort_rec terminates, the temporary array is deallocated.
 */
void mergesort(int *v, int n)
{
    int* tmp = (int*)malloc(n*sizeof(v[0]));
    assert(tmp != NULL);
#ifdef SERIAL
    /* [TODO] Create a parallel region, and make sure that only one
       thread calls mergesort_rec() to start the recursion. */
#else
#pragma omp parallel default(none) shared(v,tmp,n)
#pragma omp single
#endif
    mergesort_rec(v, 0, n-1, tmp);
    free(tmp);
}

/* Returns a random integer in the range [a..b], inclusive */
int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

/**
 * Fills a[] with a random permutation of the intergers 0..n-1; the
 * caller is responsible for allocating a
 */
void fill(int* a, int n)
{
    int i;
    for (i=0; i<n; i++) {
        a[i] = (int)i;
    }
    for (i=0; i<n-1; i++) {
        int j = randab(i, n-1);
        swap(a+i, a+j);
    }
}

/* Return 1 iff a[] contains the values 0, 1, ... n-1, in that order */
int is_correct(const int* a, int n)
{
    int i;
    for (i=0; i<n; i++) {
        if ( a[i] != i ) {
            fprintf(stderr, "Expected a[%d]=%d, got %d\n", i, i, a[i]);
            return 0;
        }
    }
    return 1;
}

int main( int argc, char* argv[] )
{
    int n = 10000000;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if (n > 1000000000) {
        fprintf(stderr, "FATAL: array too large\n");
        return EXIT_FAILURE;
    }

    int *a = (int*)malloc(n*sizeof(a[0]));
    assert(a != NULL);

    printf("Initializing array...\n");
    fill(a, n);
    printf("Sorting %d elements...", n); fflush(stdout);
    const double tstart = omp_get_wtime();
    mergesort(a, n);
    const double elapsed = omp_get_wtime() - tstart;
    printf("done\n");
    const int ok = is_correct(a, n);
    printf("Check %s\n", (ok ? "OK" : "failed"));
    printf("Elapsed time: %f\n", elapsed);

    free(a);

    return EXIT_SUCCESS;
}
