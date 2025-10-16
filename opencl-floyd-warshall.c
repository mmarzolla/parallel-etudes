/******************************************************************************
 *
 * opencl-floyd-warshall.cu - All-pair shortest paths.
 *
 * Copyright (C) 2025 Moreno Marzolla
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
% Last updated: 2025-10-16

This program computes all-pair shortest path distances on a weighted.
directed graph using Flyd and Warshall's algorithm.

The input is in [DIMACS
format](http://www.diag.uniroma1.it/challenge9/index.shtml), and is
read from standard input; the program computes all distances, and
prints to stadandard ouptut the minimum distance from node $0$ to
$n-1$, where $n$ is the number of nodes of the input.

Some test files are provided: [rome99.gr](rome99.gr) (part of the road
map of Rome), [DE.gr](DE.gr) (part of the road map of Delaware),
[VT.gr](VT.gr) (part of the road map of Vermont), [ME.gr](ME.gr) (part
of the road map of Maine) and [NV.gr](NV.gr) (part of the road map of
Nevada). The number of nodes $n$ and edges $m$ of the input graphs,
and the distances between node $0$ and $n-1$, are shown in Table 1.

:Table 1: Parameters of the input datasets.

Graph                        Nodes ($n$)    Edges ($m$)    Distance $0 \rightarrow n-1$
-------------------------  ------------- -------------- -------------------------------
[graph100.gr](graph100.gr)           100           4932                            16.0
[graph1000.gr](graph1000.gr)        1000         499623                             4.0
[graph2000.gr](graph2000.gr)        2000         800066                             4.0
[rome99.gr](rome99.g)               3353           8870                         30290.0

The goal of this exercise is to parallelize the function
`floyd_warshall()` using OpenCL. Note that the main nested loop of the
Floyd-Warshall algorithm is non embarrassingly parallel, since it has
loop-carried dependences. The serial code is written in such a way to
make the dependences more evident, and allows immediate application of
OpenMP directives according to the approach described in:

> Tang, Peiyi. "Rapid development of parallel blocked all-pairs shortest paths code for multi-core computers", proc. IEEE SOUTHEASTCON 2014, <https://doi.org/10.21122/2309-4923-2022-3-57-65>

![Figure 1: Data dependences for the Floyd-Warshall algorithm.](floyd-warshall.svg)

Specifically, the dependences for the Floyd-Warshall algorithm are
shown in Figure 1. We observe that:

1. The distance $d_{kk}$ has no dependency;
2. Distances of row and column $k$ depend on $d_{kk}$;
3. All other distances depend on rown and column $k$.

This suggests that the computation is broken into three sequential
steps:

1. During the first step, compute $d_{kk}$;
2. During the second step, compute the distances on row and column $k$,
   in parallel;
3. During the third step, compute everything else, in parallel.

## Files

- [omp-floyd-warshall.cu](omp-floyd-warshall.cu) [simpleCL.h](simpleCL.h) [simpleCL.c](simpleCL.c)

***/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
#include "hpc.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> /* for isinf(), fminf() and HUGE_VAL */
#include <assert.h>
#include "simpleCL.h"

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
#ifndef SERIAL

sclKernel kernel_fw_init1, kernel_fw_init2, kernel_fw_relax0, kernel_fw_relax1,  kernel_fw_relax2, kernel_fw_check;

int floyd_warshall( const graph_t *g, float *d, int *p )
{
    assert(g != NULL);
    int result = 0;

    const int n = g->n;
    const int m = g->m;
#if 1
    sclDim BLOCK_2D_NN, GRID_2D_NN, BLOCK_1D_N, GRID_1D_N, BLOCK_1D_M, GRID_1D_M;
    sclWGSetup2D(n, n, &GRID_2D_NN, &BLOCK_2D_NN);
    sclWGSetup1D(n, &GRID_1D_N, &BLOCK_1D_N);
    sclWGSetup1D(m, &GRID_1D_M, &BLOCK_1D_M);
#else
    const sclDim BLOCK_2D_NN = DIM2(SCL_DEFAULT_WG_SIZE2D, SCL_DEFAULT_WG_SIZE2D);
    const sclDim GRID_2D_NN = DIM2(sclRoundUp(n, SCL_DEFAULT_WG_SIZE2D),
                                   sclRoundUp(n, SCL_DEFAULT_WG_SIZE2D));
    const sclDim BLOCK_1D_N = DIM1(SCL_DEFAULT_WG_SIZE);
    const sclDim GRID_1D_N = DIM1(sclRoundUp(n, SCL_DEFAULT_WG_SIZE));
    const sclDim BLOCK_1D_M = DIM1(SCL_DEFAULT_WG_SIZE);
    const sclDim GRID_1D_M = DIM1(sclRoundUp(m, SCL_DEFAULT_WG_SIZE));
#endif
    cl_mem d_d = sclMalloc(n*n*sizeof(*d), CL_MEM_READ_WRITE);
    cl_mem d_p = sclMalloc(n*n*sizeof(*p), CL_MEM_READ_WRITE);
    cl_mem d_edges = sclMallocCopy(m*sizeof(edge_t), g->edges, CL_MEM_READ_ONLY);
    cl_mem d_result = sclMallocCopy(sizeof(result), &result, CL_MEM_READ_WRITE);

    sclSetArgsEnqueueKernel(kernel_fw_init1,
                            GRID_2D_NN, BLOCK_2D_NN,
                            ":b :b :d",
                            d_d, d_p, n);

    sclSetArgsEnqueueKernel(kernel_fw_init2,
                            GRID_1D_M, BLOCK_1D_M,
                            ":b :b :b :d :d",
                            d_edges, d_d, d_p, n, m);

    for (int k=0; k<n; k++) {
        sclSetArgsEnqueueKernel(kernel_fw_relax0,
                                DIM1(1), DIM1(1),
                                ":b :b :d :d",
                                d_d, d_p, k, n);
        sclSetArgsEnqueueKernel(kernel_fw_relax1,
                                GRID_1D_N, BLOCK_1D_N,
                                ":b :b :d :d",
                                d_d, d_p, k, n);
        sclSetArgsEnqueueKernel(kernel_fw_relax2,
                                GRID_2D_NN, BLOCK_2D_NN,
                                ":b :b :d :d",
                                d_d, d_p, k, n);
    }

    sclSetArgsEnqueueKernel(kernel_fw_check,
                            GRID_1D_N, BLOCK_1D_N,
                            ":b :d :b",
                            d_d, n, d_result);

    sclMemcpyDeviceToHost(d, d_d, n*n*sizeof(*d));
    sclMemcpyDeviceToHost(p, d_p, n*n*sizeof(*p));
    sclMemcpyDeviceToHost(&result, d_result, sizeof(result));

    sclFree(d_d);
    sclFree(d_p);
    sclFree(d_edges);

    return result;
}

#else
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

    for (int u=0; u<n; u++) {
        for (int v=0; v<n; v++) {
            d[IDX(u,v,n)] = (u == v ? 0.0 : HUGE_VAL);
            p[IDX(u,v,n)] = -1;
        }
    }

    for (const edge_t *e = g->edges; e < g->edges + m; e++) {
        d[IDX(e->src,e->dst,n)] = e->w;
        p[IDX(e->src,e->dst,n)] = e->src;
    }

    for (int k=0; k<n; k++) {
        fw_relax(d, p, k, k, k, n);

        for (int i=0; i<n; i++) {
            if (i == k) continue;
            fw_relax(d, p, k, i, k, n);
            fw_relax(d, p, i, k, k, n);
        }

        for (int u=0; u<n; u++) {
            if (u == k) continue;
            /* u != k here */
            for (int v=0; v<n; v++) {
                if (v == k) continue;
                /* u != k /\ v != k here */
                fw_relax(d, p, u, v, k, n);
            }
        }
    }

    /* Check for self-loops of negative cost. Of one is found, there
       are negative-weight cycles and return 1. */
    for (int u=0; u<n; u++) {
        if ( d[IDX(u,u,n)] < 0.0 ) {
            /* printf("d[%d][%d] = %f\n", u, u, d[u][u]); */
            return 1;
        }
    }

    return 0;
}
#endif

int main( int argc, char* argv[] )
{
    graph_t g;
    float *d;
    int *p;

    if ( argc > 1 ) {
        fprintf(stderr, "Usage: %s < problem_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    load_dimacs(stdin, &g);

    /* Care must be taken to convert g.n to (size_t) to avoid
       overflows is the number of nodes is very large. */
    d = (float*)malloc((size_t)g.n * (size_t)g.n * sizeof(*d)); assert(d);
    p = (int*)malloc((size_t)g.n * (size_t)g.n * sizeof(*p)); assert(p);

#ifndef SERIAL
    sclInitFromFile("opencl-floyd-warshall.cl");
    kernel_fw_init1 = sclCreateKernel("fw_init1");
    kernel_fw_init2 = sclCreateKernel("fw_init2");
    kernel_fw_relax0 = sclCreateKernel("fw_relax0");
    kernel_fw_relax1 = sclCreateKernel("fw_relax1");
    kernel_fw_relax2 = sclCreateKernel("fw_relax2");
    kernel_fw_check = sclCreateKernel("fw_check");
#endif
    const float tstart = hpc_gettime();
    floyd_warshall(&g, d, p);
    const float elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Execution time %.3f\n", elapsed);

    printf("d[%d,%d] = %f\n", 0, g.n-1, d[IDX(0, g.n-1, g.n)]);
    free(d);
    free(p);
    free(g.edges);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
