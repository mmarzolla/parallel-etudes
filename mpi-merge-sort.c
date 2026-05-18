/****************************************************************************
 *
 * mpi-merge-sort.c - Merge Sort with MPI
 *
 * Copyright (C) 2017--2026 Moreno Marzolla
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
% Merge Sort with MPI
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-05-18

The goal of this exercise is to implement a bottom-up, iterative
implementation of the _Merge Sort_ algorithm that operates as follows:

- Initially, rank 0 knows the content of an array `v[]` of length $n$;
  all $P$ MPI processes know the length $n$.

- Process 0 scatters `v[]` across all processes; each process receives
  $n/P$ elements of `v[]`.

- Each process sorts its chunk using an efficient serial algorithm,
  e.g., the `qsort()` C function.

- All processes cooperate to perform a tree-merge of the chunks (see
  Figure 1). The procedure involves $O(\log P)$ phases and is similar
  to tree-reduction, except that during each phase processes send
  their local chunks to partners, that perform a merge to get a larger
  chunk.  At the last phase, rank 0 performs the last merge of two
  chunks of size $n/2$ to get the sorted array.

![Figure 1: Tree-merge procedure.](mpi-merge-sort.svg)

Assumptions:

- The array length $n$ must be an integer multiple of the number of
  processes $P$;

- $P$ is a power of two.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-merge-sort.c -o mpi-merge-sort

To sort an array of length $n$:

        mpirun -n [P] ./mpi-merge-sort [n]

Example:

        mpirun -n 4 ./mpi-merge-sort 500000

## Files

- [mpi-merge-sort.c](mpi-merge-sort.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

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

int compare(const void* a, const void* b)
{
    const int ai = *(const int*)a;
    const int bi = *(const int*)b;
    if (ai < bi)
        return -1;
    else if (ai == bi)
        return 0;
    else
        return 1;
}

/**
 * Merge two sorted arrays `src1[]` and `src2[]` of length `n` into
 * buffer `dst[]` of length `2*n`.
 */
void merge(const int *src1, const int *src2, int *dst, int n)
{
    int i=0, j=0, k=0;
    while (i<n && j<n) {
        if (src1[i] <= src2[j]) {
            dst[k] = src1[i++];
        } else {
            dst[k] = src2[j++];
        }
        k++;
    }
    /* Handle leftovers */
    while (i<n) {
        dst[k] = src1[i++];
        k++;
    }
    while (j<n) {
        dst[k] = src2[j++];
        k++;
    }
}

/**
 * The entry point of the Merge-Sort algorithm. Sort array `v[]` of
 * length `b`. `v` is meaningful only at rank 0; all other processes
 * must ignore it.
 */
void mergesort(int *v, int n)
{
    int my_rank, comm_sz;

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

#ifdef SERIAL
    /* TODO: implement the bottom-up merge-sort algorithm. */
    if (my_rank == 0)
        qsort(v, n, sizeof(*v), compare);
#else
    int local_n = n / comm_sz;
    int *local_v = malloc(local_n * sizeof(*local_v));
    assert(local_v != NULL);

    /* Scatter `v` across all processes. */
    MPI_Scatter(v,              // sendbuf,
                local_n,        // sendcount
                MPI_INT,        // sendtype
                local_v,        // recvbuf
                local_n,        // recvcount
                MPI_INT,        // recvtype
                0,              // root
                MPI_COMM_WORLD);

    /* Each process sorts it local array. */
    qsort(local_v, local_n, sizeof(int), compare);

    /* Tree-merge. */
    for (int p=comm_sz; p>1; p = p/2) {

        /* Only the first `p` processes are active at each round; all
           the other ones skip the computation. */

        if (my_rank >= p)
            continue;

        int *remote_v = malloc(local_n * sizeof(*remote_v)); assert(remote_v != NULL);
        int *new_v = malloc(local_n*2 * sizeof(*new_v)); assert(new_v != NULL);

        if (my_rank < p/2) {
            MPI_Recv(remote_v,          // buf
                     local_n,           // count
                     MPI_INT,           // datatype
                     my_rank + p/2,     // source
                     MPI_ANY_TAG,       // tag
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            merge(local_v, remote_v, new_v, local_n);
            free(remote_v);
            free(local_v);
            local_v = new_v;
        } else {
            MPI_Send(local_v,           // buf
                     local_n,           // count
                     MPI_INT,           // datatype
                     my_rank - p/2,     // dest
                     0,                 // tag
                     MPI_COMM_WORLD);
        }
        local_n *= 2;
    }
    if (my_rank == 0)
        memcpy(v, local_v, n*sizeof(*v));
    free(local_v);
#endif
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
    for (int i=0; i<n; i++) {
        a[i] = (int)i;
    }
    for (int i=0; i<n-1; i++) {
        const int j = randab(i, n-1);
        swap(a+i, a+j);
    }
}

/* Return 1 iff `a[]` contains 0, 1, ... n-1, in that order. */
int is_correct(const int* a, int n)
{
    for (int i=0; i<n; i++) {
        if ( a[i] != i ) {
            fprintf(stderr, "Expected a[%d]=%d, got %d\n", i, i, a[i]);
            return 0;
        }
    }
    return 1;
}

/* Return 1 iff `n` is a power of two. */
int is_power_of_two(int n)
{
    return (n & (n-1)) == 0;
}

int main( int argc, char* argv[] )
{
    int my_rank, comm_sz;
    int *a = NULL;
    double tstart, elapsed;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int n = 10000000;

    if ( argc > 2 && my_rank == 0) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if (my_rank == 0) {
        if (n > 1000000000) {
            fprintf(stderr, "FATAL: array too large\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (!is_power_of_two(comm_sz)) {
            fprintf(stderr, "FATAL: the number of processors must be a power of two\nb");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (n % comm_sz) {
            fprintf(stderr, "FATAL: the array length must be a multiple of the number of processors\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        a = (int*)malloc(n*sizeof(a[0]));
        assert(a != NULL);

        printf("Initializing array...\n");
        fill(a, n);
        printf("Sorting %d elements...\n", n);
        tstart = MPI_Wtime();
    }
    mergesort(a, n);
    if (my_rank == 0) {
        elapsed = MPI_Wtime() - tstart;
        printf("done\n");
        const int ok = is_correct(a, n);
        printf("Check %s\n", (ok ? "OK" : "failed"));
        printf("Execution time %.3f\n", elapsed);
    }
    free(a);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
