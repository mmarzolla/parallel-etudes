/****************************************************************************
 *
 * omp-list-ranking.c - Parallel list ranking
 *
 * Copyright (C) 2021, 2022, 2024 Moreno Marzolla
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
% Last updated: 2024-09-24

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
   time.

   Upon termination, `ranks[i]` is the rank of `nodes[i]`. */
void rank( list_node_t *nodes, int start, int *ranks, int n )
{
#ifdef SERIAL
    int rank = n;
    for (i = start; i >= 0; i = nodes[i].next) {
        ranks[i] = --rank;
    }
#else
    int done = 0;
    int *cur_rank = (int*)malloc(n * sizeof(*cur_rank)); assert(cur_rank);
    int *new_rank = (int*)malloc(n * sizeof(*new_rank)); assert(new_rank);
    int *cur_next = (int*)malloc(n * sizeof(*cur_next)); assert(cur_next);
    int *new_next = (int*)malloc(n * sizeof(*new_next)); assert(new_next);
    assert(ranks);

    /* Initialization */
#pragma omp parallel for default(none) shared(nodes,cur_rank,cur_next,n)
    for (int i=0; i<n; i++) {
        if (nodes[i].next < 0)
            cur_rank[i] = 0;
        else
            cur_rank[i] = 1;
        cur_next[i] = nodes[i].next;
    }

    /* Compute ranks */
    while (!done) {
        done = 1;
#pragma omp parallel default(none) shared(done,n,nodes,cur_rank,cur_next,new_rank,new_next)
        {
#pragma omp for
            for (int i=0; i<n; i++) {
                if (cur_next[i] >= 0) {
                    done = 0;
                    new_rank[i] = cur_rank[i] + cur_rank[cur_next[i]];
                    new_next[i] = cur_next[cur_next[i]];
                } else {
                    new_rank[i] = cur_rank[i];
                    new_next[i] = cur_next[i];
                }
            }
            /* Swap cur and next */
#pragma omp single
            {
                int *tmp;
                tmp = cur_rank;
                cur_rank = new_rank;
                new_rank = tmp;

                tmp = cur_next;
                cur_next = new_next;
                new_next = tmp;
            }
        }
    }
    memcpy(ranks, cur_rank, n * sizeof(*ranks));
    free(cur_rank);
    free(new_rank);
    free(cur_next);
    free(new_next);
#endif
}

/* Inizializza il contenuto della lista. Per agevolare il controllo di
   correttezza, il valore presente in ogni nodo coincide con il rango
   che ci aspettiamo venga calcolato. */
void init(list_node_t *nodes, int n, int *start)
{
    for (int i=0; i<n; i++) {
        nodes[i].val = n-1-i;
        nodes[i].rank = -1;
        nodes[i].next = (i+1<n ? i + 1 : -1);
    }
    *start = 0;
}

/* Controlla la correttezza del risultato */
int check(const list_node_t *nodes, const int *ranks, int n)
{
    for (int i=0; i<n; i++) {
        if (ranks[i] != nodes[i].val) {
            fprintf(stderr, "FAILED: rank[%d]=%d, expected %d\n", i, ranks[i], nodes[i].val);
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
    int *ranks = (int*)malloc(n * sizeof(*ranks));
    assert(nodes != NULL);
    init(nodes, n, &start);
    rank(nodes, start, ranks, n);
    check(nodes, ranks, n);
    free(nodes);
    return EXIT_SUCCESS;
}
