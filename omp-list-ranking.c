/****************************************************************************
 *
 * omp-list-ranking.c - Parallel list ranking
 *
 * Copyright (C) 2021, 2022, 2024 by Moreno Marzolla <moreno.marzolla@unibo.it>
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
% HPC - Parallel list ranking
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2024-09-24

The goal of this exercise is to implement the _list ranking_
algorithm. The algorithm takes a list of length $n$ as input. Each
node contains the following attributes:

- An arbitrary value `val`, representing some information stored at
  each node;

- An integer `rank`, initially undefined;

- A pointer to the next element of the list (or `NULL`, if the node has
  no successor).

Upon termination, the algorithm must set the `rank` atribute of a node
to its distance (number of links) to the _end_ of the list. Therefore,
the last node has `rank = 0`, the previous one has `rank = 1`, and so
forth up to the head of the list that has `rank = n-1`. It is not
required that the algorithm keeps the original values of the `next`
attribute, i.e., upon termination the relationships between nodes may
be undefined.

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

First of all, right before the `While` cycle there must be a barrier
synchronization so that all distances are properly initialized before
the actual pointer jumping algorithm starts.

Then, the pseudocode above assumes that all instructions are executed
in a SIMD way, which is something that does not happen with OpenMP.
In particular, the instruction

```
Set d[n] ← d[n] + d[n.next].
```

has a loop-carried dependency on `d[]`. Indeed, the pseudocode assumes
that all processors _first_ compute `d[n] + d[n.next]`, and _then, all
at the same time_, set the new value of `d[n]`.

![Figure 1: Pointer jumping algorithm](omp-list-ranking.svg)

To compile:

        gcc -std=c99 -fopenmp -Wall -Wpedantic omp-list-ranking.c -o omp-list-ranking

To execute:

        ./omp-list-ranking [n]

where `n` is the length of the list.

For example, to execute with $P=4$ OpenMP threads and $n = 1000$
nodes:

        OMP_NUM_THREADS=4 ./omp-list-ranking 1000

> **Note** The list ranking algorithm requires that each thread has
> direct access to some node(s) of the list (it does not matter which
> nodes). To allow $O(1)$ access time, nodes are stored in an array of
> length $n$. Note that the first element of the array is _not_
> necessarily the head of the list, and element at position $i+1$ is
> _not_ necessarily the successor of element at posizion $i$.

## Files

- [omp-list-ranking.c](omp-list-ranking.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <omp.h>

typedef struct list_node_t {
    int val;    /* arbitrary value at this node */
    int rank;   /* rank of this node            */
    struct list_node_t *next;
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

/* Compute the rank of the `n` nodes in the array `nodes`.  Note that
   the array contains nodes that are connected in a singly linked-list
   fashion, so `nodes[0]` is not necessarily the head of the list, and
   `nodes[i+1]` is not necessarily the successor of `nodes[i]`. The
   array serves only as a conveniente way to allow each OpenMP thread
   to grab an element of the list in constant time.

   Upon return, all nodes have their `rank` field correctly set. Note
   that the `next` field will be set to NULL, hence the structure of
   the list will essentially be destroyed. This could be avoided with
   a bit more care. */
void rank( list_node_t *nodes, int n )
{
    int done = 0;
    int *new_rank = (int*)malloc(n * sizeof(*new_rank));
    list_node_t **new_next = (list_node_t**)malloc(n * sizeof(*new_next));

    /* initialize ranks */
#pragma omp parallel for default(none) shared(nodes,n)
    for (int i=0; i<n; i++) {
        if (nodes[i].next == NULL)
            nodes[i].rank = 0;
        else
            nodes[i].rank = 1;
    }

    /* compute ranks */
    while (!done) {
        done = 1;
#pragma omp parallel default(none) shared(done,n,nodes,new_rank,new_next)
        {
#pragma omp for
            for (int i=0; i<n; i++) {
                if (nodes[i].next != NULL) {
                    done = 0; // not a real race condition
                    new_rank[i] = nodes[i].rank + nodes[i].next->rank;
                    new_next[i] = nodes[i].next->next;
                } else {
                    new_rank[i] = nodes[i].rank;
                    new_next[i] = nodes[i].next;
                }
            }
            /* Update ranks */
#pragma omp for
            for (int i=0; i<n; i++) {
                nodes[i].rank = new_rank[i];
                nodes[i].next = new_next[i];
            }
        }
    }
    free(new_rank);
    free(new_next);
}

/* Inizializza il contenuto della lista. Per agevolare il controllo di
   correttezza, il valore presente in ogni nodo coincide con il rango
   che ci aspettiamo venga calcolato. */
void init(list_node_t *nodes, int n)
{
    for (int i=0; i<n; i++) {
        nodes[i].val = n-1-i;
        nodes[i].rank = -1;
        nodes[i].next = (i+1<n ? nodes + (i + 1) : NULL);
    }
}

/* Controlla la correttezza del risultato */
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

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    list_node_t *nodes = (list_node_t*)malloc(n * sizeof(*nodes));
    assert(nodes != NULL);
    init(nodes, n);
    rank(nodes, n);
    check(nodes, n);
    free(nodes);
    return EXIT_SUCCESS;
}
