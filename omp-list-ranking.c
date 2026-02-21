/****************************************************************************
 *
 * omp-list-ranking.c - Parallel list ranking
 *
 * Copyright (C) 2021, 2022, 2024, 2026 Moreno Marzolla
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
% Parallel list ranking
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-02-21

The goal of this exercise is to implement the _list ranking_
algorithm. The algorithm takes a list of length $n$ as input; the list
is encoded into an array of length `n` of items of type
`list_node_t`. Each node contains the following attributes:

- An arbitrary value `val`, representing some information stored at
  each node;

- An integer `rank`, initially undefined;

- An integer `next` that is the index of the successor, or -1 if there
  is no successor..

Upon termination, the algorithm must set the `rank` of each node to
its distance to the _end_ of the list: the last node has `rank = 0`,
the previous one has `rank = 1`, and so forth up to the head of the
list that has `rank = n-1`. Upon termination, the algorithm must
preserve the `val` and `next` attributes of all nodes of the list.

List ranking can be implemented using a technique called _pointer
jumping_. The following pseudocode (source:
<https://en.wikipedia.org/wiki/Pointer_jumping>) shows a possible
implementation, with a few caveats described below.

```
Allocate an array of N integers.
Initialize: for each processor/list node n, in parallel:
   If n.next = nil, set d[n] ← 0.
      Else, set d[n] ← 1.
   While any node n has n.next ≠ nil:
      For each processor/list node n, in parallel:
         If n.next ≠ nil:
             Set d[n] ← d[n] + d[n.next].
             Set n.next ← n.next.next.
```

First of all, right before the `While` loop there must be a barrier
synchronization so that all distances are properly initialized before
the actual pointer jumping algorithm starts.

Then, the pseudocode assumes that all instructions are executed in a
SIMD way, which is something that does not happen with OpenMP.  In
particular, the instruction

```
Set d[n] ← d[n] + d[n.next].
```

has a loop-carried dependence on `d[]`. Indeed, the pseudocode assumes
that all processors _first_ compute `d[n] + d[n.next]`, and _then, all
at the same time_, set the new value of `d[n]`.

![Figure 1: Pointer jumping algorithm.](omp-list-ranking.svg)

To compile:

        gcc -std=c99 -fopenmp -Wall -Wpedantic omp-list-ranking.c -o omp-list-ranking

To execute:

        ./omp-list-ranking [n]

where `n` is the length of the list.

For example, to execute with $P=4$ OpenMP threads and $n = 1000$
nodes:

        OMP_NUM_THREADS=4 ./omp-list-ranking 1000

## Files

- [omp-list-ranking.c](omp-list-ranking.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <omp.h>

typedef struct list_node_t {
    int val;    /* arbitrary value at this node                 */
    int rank;   /* rank of this node                            */
    int next;   /* Position of the next element; -1 if null     */
} list_node_t;

/* Print the content of array `nodes` of length `n` */
void list_print(const list_node_t *nodes, int n)
{
    printf("**\n** list content\n**\n");
    for (int i=0; i<n; i++) {
        printf("[%d] val=%d rank=%d\n", i, nodes[i].val, nodes[i].rank);
    }
    printf("\n");
}

/* Compute the rank of the `n` nodes in the array `nodes`.  The nodes
   in the array are connected as a linked-list; the first node of the
   list is `nodes[start]`; the successor of `nodes[i]` is
   `nodes[nodes[i].next]`. The array serves as a conveniente way to
   allow each OpenMP thread to grab an element of the list in constant
   time. */
void rank( list_node_t *nodes, int start, int n )
{
#ifdef SERIAL
    int rank = n;
    for (i = start; i >= 0; i = nodes[i].next) {
        ranks[i] = --rank;
    }
#else
    int done = 0;
    int *rank[2], *next[2];
    rank[0] = (int*)malloc(n * sizeof(int)); assert(rank[0]);
    rank[1] = (int*)malloc(n * sizeof(int)); assert(rank[1]);
    next[0] = (int*)malloc(n * sizeof(int)); assert(next[0]);
    next[1] = (int*)malloc(n * sizeof(int)); assert(next[1]);
    int cur = 0, new = 1;

    /* The following macros make the code more readable */
#define CUR_RANK(i) rank[cur][i]
#define NEW_RANK(i) rank[new][i]
#define CUR_NEXT(i) next[cur][i]
#define NEW_NEXT(i) next[new][i]

    /* Initialization */
#pragma omp parallel for default(none) shared(nodes,rank,next,n,cur)
    for (int i=0; i<n; i++) {
        if (nodes[i].next < 0)
            CUR_RANK(i) = 0;
        else
            CUR_RANK(i) = 1;
        CUR_NEXT(i) = nodes[i].next;
    }

    /* Compute ranks */
    while (!done) {
        done = 1;
#pragma omp parallel for default(none) shared(n, done, rank, next, cur, new)
        for (int i=0; i<n; i++) {
            if (CUR_NEXT(i) >= 0) {
                done = 0;
                NEW_RANK(i) = CUR_RANK(i) + CUR_RANK(CUR_NEXT(i));
                NEW_NEXT(i) = CUR_NEXT(CUR_NEXT(i));
            } else {
                NEW_RANK(i) = CUR_RANK(i);
                NEW_NEXT(i) = CUR_NEXT(i);
            }
        }
        /* Swap cur and next */
        cur = 1 - cur;
        new = 1 - cur;
    }

#pragma omp parallel for
    for (int i=0; i<n; i++) {
        nodes[i].rank = CUR_RANK(i);
    }

#undef CUR_RANK
#undef NEW_RANK
#undef CUR_NEXT
#undef NEW_NEXT

    free(rank[0]); free(rank[1]);
    free(next[0]); free(next[1]);
#endif
}

/* Initialize the list. To simplify the correctness test, the value of
   each node is its rank. */
void init(list_node_t *nodes, int n, int *start)
{
    for (int i=0; i<n; i++) {
        nodes[i].val = n-1-i;
        nodes[i].rank = -1;
        nodes[i].next = (i+1<n ? i + 1 : -1);
    }
    *start = 0;
}

/* Check result. */
int check(const list_node_t *nodes, int n)
{
    for (int i=0; i<n; i++) {
        if (nodes[i].rank != nodes[i].val) {
            fprintf(stderr, "FAILED: rank[%d]=%d, expected %d\n", i, nodes[i].rank, nodes[i].val);
            return 0;
        }
    }
    fprintf(stderr, "Check OK\n");
    return 1;
}

int main( int argc, char *argv[] )
{
    int n = 1000;
    int start;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    list_node_t *nodes = (list_node_t*)malloc(n * sizeof(*nodes));
    assert(nodes != NULL);
    init(nodes, n, &start);
    rank(nodes, start, n);
    check(nodes, n);
    free(nodes);
    return EXIT_SUCCESS;
}
