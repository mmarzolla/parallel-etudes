/******************************************************************************
 *
 * omp-floyd-warshall.c - All-pair shortest paths.
 *
 * Copyright (C) 2024--2026 Moreno Marzolla
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
 *****************************************************************************/

/***
% All-pair shortest paths
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-04-30

The file [omp-floyd-warshall.c](omp-floyd-warshall.c) contains a
serial implementation of [Floyd and Warshall's
algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)
for computing shortest-path distances among all pair of nodes on a
directed, weighted graph.

Floyd-Warshall's algorithm uses dynamic programming. Let $d_{uv}^k$
the (shortest) distance from node $u$ to node $j$, $0 \leq u,v< n$,
with the assumption that all intermediate nodes (if any) must belong
to the set $\{0, \ldots, k\}$; $d_{uv}^{-1}$ is the shortest-path
distance from $u$ to $v$ that does not visit intermediate
nodes. Therefore, we have:

$$
d_{uv}^{-1} = \begin{cases}
0 & \text{if}\ u = v \\
w(u,v) & \text{otherwise}
\end{cases}
$$

where $w(u,v)$ is the weight of edge $(u,v)$; if no such edge exists,
then $w(u,v) = +\infty$.

For each $k=0, \ldots, n-1$ we can define $d_{uv}^k$ recursively as
follows:

$$
d_{uv}^k = \min \left\lbrace d_{uv}^{k-1} , d_{uk}^{k-1} + d_{kv}^{k-1} \right\rbrace
$$

the idea being that the shortest path from $u$ to $v$ that only visits
intermediate nodes in $\{0, \ldots, k\}$ can either pass through node
$k$ or not:

- If the shortest path does not pass through node $k$, then its
  length is $d_{uv}^{k-1}$;

- If the shortest path does pass through node $k$, then it can be
  broken into two sub-paths, the first one from $u$ to $k$, and the
  second one from $k$ to $v$. By construction, those sub-paths will
  use intermediate nodes in $\{0, \ldots, k-1\}$ only, since shortest
  paths can not have cycles and therefore node $k$ can only be visited
  once. So, the sum of their lengths is $d_{uk}^{k-1} + d_{kv}^{k-1}$.

It can be shown that we can drop the subscript and compute the
distances $d_{uv}$ iteratively using the following Java (pseudo-)code:

```Java
// Return true if there are negative-weight cycles
boolean floyd_warshall(double d[][], Graph G) {
   final int n = G.n; // number of nodes

   // Initialization
   for (int u=0; u<n; u++) {
     for (int v=0; v<n; v++) {
       d[u][v] = (u == v ? 0 : Double.POSITIVE_INFINITY);
     }
   }

   for (Edge e: G.edges()) {
     d[e.src][e.dst] = e.w;
   }

   // k relaxation phases
   for (int k=0; k<n; k++) {
     for (int u=0; u<n; u++) {
       for (int v=0; v<n; v++) {
         if (d[u][k] + d[k][v] < d[u][v])
           d[u][v] = d[u][k] + d[k][v];
       }
     }
   }

   // check for negative-weight cycles
   for (int u=0; u<n; u++) {
     if ( d[u][u] < 0 ) {
       return true;
     }
   }
   return false;
}
```

The program reads input in [DIMACS
format](http://www.diag.uniroma1.it/challenge9/index.shtml) from
standard input; at the end, it computes all distances and prints to
stadandard ouptut the minimum distance from node $0$ to $n-1$, where
$n$ is the number of nodes of the input.

Some test files are provided; Table 1 shows the number of nodes $n$
and edges $m$ of each one, plus the distances between node $0$ and
$n-1$.

:Table 1: Parameters of the input datasets.

Graph                             Nodes ($n$)    Edges ($m$)    Distance $0 \rightarrow n-1$
------------------------------  ------------- -------------- -------------------------------
[graph10.gr](graph10.gr)                   10             52                            41.0
[graph100.gr](graph100.gr)                100           4932                            16.0
[graph1000.gr](graph1000.gr)             1000         499623                             4.0
[graph2000.gr](graph2000.gr)             2000         800066                             4.0
[rome99.gr](rome99.g)                    3353           8870                         30290.0

The goal of this exercise is to parallelize the function
`floyd_warshall()` using OpenMP. The main nested loop of the
Floyd-Warshall algorithm has loop-carried dependences, so it can not
be trivially parallelized. However, in the provided serial code we
have broken down the loop into three sequential phases, each phase
being embarrassingly parallel[^1].

[^1]: See Tang, Peiyi. _Rapid development of parallel blocked all-pairs shortest paths code for multi-core computers_, proc. IEEE SOUTHEASTCON 2014, <https://doi.org/10.1109/SECON.2014.6950734>

![Figure 1: Data dependences for the Floyd-Warshall algorithm.](floyd-warshall.svg)

The dependences for the Floyd-Warshall algorithm are shown in Figure
1. We observe that:

1. The distance $d_{kk}$ has no dependence;
2. Distances of row and column $k$ depend on $d_{kk}$;
3. All other distances depend on row and column $k$.

This suggests that the computation is broken into three steps:

1. Compute $d_{kk}$;
2. Compute the distances on row and column $k$, in parallel;
3. Compute everything else, in parallel.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-floyd-warshall.c -o omp-floyd-warshall

Execute with:

        ./omp-floyd-warshall graph100.gr

## Files

- [omp-floyd-warshall.c](omp-floyd-warshall.c)

***/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> /* for isinf(), fminf() and HUGE_VALF */
#include <assert.h>
#include <omp.h>

typedef struct {
    int src, dst;
    float w;
} edge_t;

/* A graph is represented as an array of edges. */
typedef struct {
    int n; /* number of nodes */
    int m; /* length of the edges array */
    edge_t *edges; /* array of edges */
} graph_t;

/**
 * Load a graph description in DIMACS format from file `f`; store the
 * graph in `g` (the caller is responsible for providing a pointer to
 * an already allocated struct). This function interprets `g` as a
 * directed graph. For more information on the DIMACS challenge format
 * see <http://www.diag.uniroma1.it/challenge9/index.shtml>.
 */
void load_dimacs(FILE *f, graph_t* g)
{
    const size_t BUFLEN = 1024;
    char buf[BUFLEN], prob[BUFLEN];
    int n, m, src, dst, w;
    int idx = 0; /* index in the edge array */
    int nmatch;

    while ( fgets(buf, BUFLEN, f) ) {
        switch( buf[0] ) {
        case 'c':
            break; /* ignore comment lines */
        case 'p':
            /* Parse problem format; expect type "shortest path" */
            sscanf(buf, "%*c %s %d %d", prob, &n, &m);
            if (strcmp(prob, "sp")) {
                fprintf(stderr, "FATAL: unknown DIMACS problem type %s\n", prob);
                exit(EXIT_FAILURE);
            }
            fprintf(stderr, "DIMACS %s with %d nodes and %d edges\n", prob, n, m);
            g->n = n;
            g->m = m;
            g->edges = (edge_t*)malloc((g->m)*sizeof(edge_t)); assert(g->edges);
            idx = 0;
            break;
        case 'a':
            nmatch = sscanf(buf, "%*c %d %d %d", &src, &dst, &w);
            if (nmatch != 3) {
                fprintf(stderr, "FATAL: Malformed line \"%s\"\n", buf);
                exit(EXIT_FAILURE);
            }
            /* In the DIMACS format, nodes are numbered starting from
               1; we use zero-based indexing internally, so we
               decrement the ids by one */
            src--;
            dst--;
            assert(src >= 0 && src < g->n);
            assert(dst >= 0 && dst < g->n);
            assert(idx < g->m);
            g->edges[idx].src = src;
            g->edges[idx].dst = dst;
            g->edges[idx].w = w;
            idx++;
            break;
        default:
            fprintf(stderr, "FATAL: unrecognized character %c on line \"%s\"\n", buf[0], buf);
            exit(EXIT_FAILURE);
        }
    }
    assert( idx == g->m );
}

/**
 * This function is used to simplify indexing of a matrix stored as a
 * linear array. Specifically, a matrix `A` with `width` columns and
 * `height` rows is stored in memory as an array of length `width *
 * height`. Element at coordinates (i,j) is at position `i * width +
 * j` of the array. To access this element, you simply write
 * `A[IDX(i,j,width)]`.
 */
int IDX(int i, int j, int width)
{
    assert((i >= 0) && (i < width));
    assert((j >= 0) && (j < width));
    return i * width + j;
}

void fw_relax(float *d, int *p, int u, int v, int k, int n)
{
    if (d[IDX(u,k,n)] + d[IDX(k,v,n)] < d[IDX(u,v,n)]) {
        d[IDX(u,v,n)] = d[IDX(u,k,n)] + d[IDX(k,v,n)];
        p[IDX(u,v,n)] = p[IDX(k,v,n)];
    }
}

/**
 * The Floyd-Warshall algorithm for all-pair shortest paths.  `g` is
 * the input graph with `n` nodes and `m` edges. `d` is the matrix of
 * distances, represented as an array of length `n * n` that must be
 * allocated by the caller; `d[IDX(u,v,n)]` is the minimum distance
 * from node `u` to node `v`. `p` is the matrix of predecessors,
 * represented as an array of length `n * n` that must be allocated by
 * the caller; `p[IDX(u,v,n)]` is the index of the node that precedes
 * `v` on the shortest path from `u` to `v`.
 *
 * Returns 1 if there are cycles of negative weights (in this case,
 * some shortest paths do not exists), 0 otherwise.
 */
int floyd_warshall( const graph_t *g, float *d, int *p )
{
    assert(g != NULL);

    const int n = g->n;
    const int m = g->m;

#ifndef SERIAL
#pragma omp parallel default(none) shared(g, d, p, n, m)
    {
#pragma omp for
#endif
    for (int u=0; u<n; u++) {
        for (int v=0; v<n; v++) {
            d[IDX(u,v,n)] = (u == v ? 0.0f : HUGE_VALF);
            p[IDX(u,v,n)] = -1;
        }
    }

#ifndef SERIAL
#pragma omp for
#endif
    for (const edge_t *e = g->edges; e < g->edges + m; e++) {
        d[IDX(e->src,e->dst,n)] = e->w;
        p[IDX(e->src,e->dst,n)] = e->src;
    }

    for (int k=0; k<n; k++) {
        /* 1. compute d_{kk}. */
#ifndef SERIAL
#pragma omp single
#endif
        fw_relax(d, p, k, k, k, n);

        /* 2. compute d_{ik} and d_{ki}, except d_{kk}. */
#ifndef SERIAL
#pragma omp for schedule(static)
#endif
        for (int i=0; i<n; i++) {
            if (i != k) {
                fw_relax(d, p, k, i, k, n);
                fw_relax(d, p, i, k, k, n);
            }
        }

        /* 3. compute everything else, except row k and column k. */
#ifndef SERIAL
#pragma omp for schedule(static)
#endif
        for (int u=0; u<n; u++) {
            if ( u != k ) {
                for (int v=0; v<n; v++) {
                    if (v != k) {
                        /* (u != k) && (v != k) here */
                        fw_relax(d, p, u, v, k, n);
                    }
                }
            }
        }
    }
#ifndef SERIAL
    } /* pragma omp parallel */
#endif
    /* Check for cycles of negative cost. Return 1 if one is found. */
    for (int u=0; u<n; u++) {
        if ( d[IDX(u,u,n)] < 0 ) {
            /* printf("d[%d][%d] = %f\n", u, u, d[u][u]); */
            return 1;
        }
    }

    return 0;
}

/* Print the path (sequence of nodes) of minimum cost from node `src`
   to `dst`. */
void print_path(const int *p, int n, int src, int dst)
{
    if (src == dst) {
        printf("%d", src);
    } else if (p[IDX(src,dst,n)] < 0) {
        printf("Not reachable");
    } else {
        print_path(p, n, src, p[IDX(src,dst,n)]);
        printf("->%d", dst);
    }
}

/* Print distances and shortest paths. */
void print_results( const float *d, const int *p, int n )
{
    printf("   s    d         dist path\n");
    printf("---- ---- ------------ -------------------------\n");
    for (int u=0; u<n; u++) {
        for (int v=0; v<n; v++) {
            printf("%4d %4d %12.4f ", u, v, d[IDX(u,v,n)]);
            print_path(p, n, u, v);
            printf("\n");
        }
        printf("---- ---- ------------ -------------------------\n");
    }
}

int main( int argc, char* argv[] )
{
    graph_t g;
    float *d;
    int *p;
    FILE *filein;

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s problem_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ((filein = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Can not open file \"%s\"\n", argv[1]);
        return EXIT_FAILURE;
    }

    load_dimacs(filein, &g);

    fclose(filein);

    /* Care must be taken to convert g.n to (size_t) to avoid
       overflows is the number of nodes is very large. */
    d = (float*)malloc((size_t)g.n * (size_t)g.n * sizeof(*d)); assert(d);
    p = (int*)malloc((size_t)g.n * (size_t)g.n * sizeof(*p)); assert(p);

    const double tstart = omp_get_wtime();
    floyd_warshall(&g, d, p);
    const double elapsed = omp_get_wtime() - tstart;

    if (g.n <= 10) {
        print_results(d, p, g.n);
    } else {
        /* too many nodes, only print distances from 0 to n-1. */
        printf("d[%d,%d] = %f\n", 0, g.n-1, d[IDX(0, g.n-1, g.n)]);
    }

    fprintf(stderr, "Execution time %.3f\n", elapsed);

    free(d);
    free(p);
    free(g.edges);
    return EXIT_SUCCESS;
}
