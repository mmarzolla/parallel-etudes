/******************************************************************************
 *
 * cuda-floyd-warshall.cu - All-pair shortest paths.
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
% Last updated: 2025-10-15

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
[rome99.gr](rome99.g)               3353           8870                         30290.0
[DE.gr](DE.gr)                     49109         121024                         69204.0
[VT.gr](VT.gr)                     97975         215116                        129866.0
[ME.gr](ME.gr)                    194505         429842                        108545.0
[NV.gr](NV.gr)                    261155         622086                        188894.0

I suggest to start experimenting using [rome99.gr](rome99.gr) because
it is the smaller one; processing the data from Nevada might require
some time, depending on the hardware.

The goal of this exercise is to parallelize the function
`floyd_warshall()` using CUDA. Note that the main nested loop of the
Floyd-Warshall algorithm is non embarrassingly parallel, since it has
loop-carried dependences. The serial code is written in such a way to
make the dependences more evident, and allows immediate application of
OpenMP directives according to the approach described in:

> Tang, Peiyi. "Rapid development of parallel blocked all-pairs shortest paths code for multi-core computers", proc. IEEE SOUTHEASTCON 2014, <https://doi.org/10.21122/2309-4923-2022-3-57-65>

## Files

- [omp-floyd-warshall.cu](omp-floyd-warshall.cu)
- Input data: [rome99.gr](rome99.gr), [DE.gr](DE.gr), [VT.gr](VT.gr), [ME.gr](ME.gr), [NV.gr](NV.gr)

***/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> /* for isinf(), fminf() and HUGE_VAL */
#include <assert.h>
#include "hpc.h"

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
__host__ __device__
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
#ifndef HANDOUT

#define BLKDIM2D 32
#define BLKDIM1D 1024

__global__
void fw_init1( float *d, int *p, int n )
{
    const int u = threadIdx.y + blockIdx.y * blockDim.y;
    const int v = threadIdx.x + blockIdx.x * blockDim.x;

    if (u<n && v<n) {
        d[IDX(u,v,n)] = (u == v ? 0.0 : HUGE_VAL);
        p[IDX(u,v,n)] = -1;
    }
}

__global__
void fw_init2(const edge_t *e, float *d, int *p, int n, int m)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < m) {
        d[IDX(e[i].src,e[i].dst,n)] = e[i].w;
        p[IDX(e[i].src,e[i].dst,n)] = e[i].src;
    }
}

__device__
void fw_relax(float *d, int *p, int u, int v, int k, int n)
{
    if (d[IDX(u,k,n)] + d[IDX(k,v,n)] < d[IDX(u,v,n)]) {
        d[IDX(u,v,n)] = d[IDX(u,k,n)] + d[IDX(k,v,n)];
        p[IDX(u,v,n)] = p[IDX(k,v,n)];
    }
}

/* Executed by one thread only; relax (k,k). */
__global__
void fw_relax0(float *d, int *p, int k, int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        fw_relax(d, p, k, k, k, n);
}

/* Executed by n threads; relax (k, *) and (*, k). */
__global__
void fw_relax1(float *d, int *p, int k, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n && i != k) {
        fw_relax(d, p, i, k, k, n);
        fw_relax(d, p, k, i, k, n);
    }
}

/* Executed by n x n threads; relax everything else. */
__global__
void fw_relax2(float *d, int *p, int k, int n)
{
    const int v = threadIdx.x + blockIdx.x * blockDim.x;
    const int u = threadIdx.y + blockIdx.y * blockDim.y;
    if (u < n && v < n && u != k && v != k)
        fw_relax(d, p, u, v, k, n);
}

__global__
void fw_check(float *d, int n, int *result)
{
    const int u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u<n) {
        if ( d[IDX(u,u,n)] < 0.0 ) {
            // no need to protect against race conditions here
            *result = 1;
        }
    }
}

int floyd_warshall( const graph_t *g, float *d, int *p )
{
    assert(g != NULL);
    float *d_d;
    int *d_p;
    edge_t *d_edges;
    int result = 0;
    int *d_result;

    const int n = g->n;
    const int m = g->m;
    const dim3 BLOCK_2D_NN(BLKDIM2D, BLKDIM2D);
    const dim3 GRID_2D_NN((n + BLKDIM2D-1)/BLKDIM2D, (n + BLKDIM2D-1)/BLKDIM2D);
    const dim3 BLOCK_1D_N(BLKDIM1D);
    const dim3 GRID_1D_N((n + BLKDIM1D-1)/BLKDIM1D);
    const dim3 BLOCK_1D_M(BLKDIM1D);
    const dim3 GRID_1D_M((m + BLKDIM1D-1)/BLKDIM1D);

    cudaSafeCall(cudaMalloc((void**)&d_d, n * n * sizeof(*d_d)) );
    cudaSafeCall(cudaMalloc((void**)&d_p, n * n * sizeof(*d_p)) );
    cudaSafeCall(cudaMalloc((void**)&d_edges, m * sizeof(*d_edges)) );
    cudaSafeCall(cudaMalloc((void**)&d_result, sizeof(*d_result)) );

    cudaSafeCall(cudaMemcpy(d_d, d, n*n*sizeof(*d_d), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_p, p, n*n*sizeof(*d_p), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_edges, g->edges, m*sizeof(*d_edges), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_result, &result, sizeof(result), cudaMemcpyHostToDevice));

    fw_init1<<<GRID_2D_NN, BLOCK_2D_NN>>>(d_d, d_p, n); cudaCheckError();

    fw_init2<<<GRID_1D_M, BLOCK_1D_M>>>(d_edges, d_d, d_p, n, m); cudaCheckError();

    for (int k=0; k<n; k++) {
        fw_relax0<<<1, 1>>>(d_d, d_p, k, n); cudaCheckError();
        fw_relax1<<<GRID_1D_N, BLOCK_1D_N>>>(d_d, d_p, k, n); cudaCheckError();
        fw_relax2<<<GRID_2D_NN, BLOCK_2D_NN>>>(d_d, d_p, k, n); cudaCheckError();
    }

    fw_check<<<GRID_1D_N, BLOCK_1D_N>>>(d_d, n, d_result); cudaCheckError();

    cudaSafeCall(cudaMemcpy(d, d_d, n*n*sizeof(*d), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(p, d_p, n*n*sizeof(*p), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(&result, d_result, sizeof(result), cudaMemcpyDeviceToHost));

    cudaFree(d_d);
    cudaFree(d_p);
    cudaFree(d_edges);

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

    const float tstart = hpc_gettime();
    floyd_warshall(&g, d, p);
    const float elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Execution time %.3f\n", elapsed);

    printf("d[%d,%d] = %f\n", 0, g.n-1, d[IDX(0, g.n-1, g.n)]);
    free(d);
    free(p);
    free(g.edges);
    return EXIT_SUCCESS;
}
