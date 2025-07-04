/******************************************************************************
 *
 * omp-floyd-warshall.c - All-pair shortest paths
 *
 * Copyright (C) 2024 Moreno Marzolla
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
% Last updated: 2024-05-09

Il programma legge da standard input una descrizione del grafo in
formato DIMACS. Vengono forniti alcuni esempi: [rome99.gr](rome99.gr)
(porzione di mappa stradale di Roma), [DE.gr](DE.gr) (porzione di
mappa del Delaware), [VT.gr](VT.gr) (porzione di mappa del Vermont),
[ME.gr](ME.gr) (porzione di mappa del Maine) e [NV.gr](NV.gr)
(porzione di mappa del Nevada). Suggerisco di iniziare con
[rome99.gr](rome99.gr) perché rappresenta il grafo più
piccolo. L'elaborazione del grafo del Nevada richiede molto tempo. Al
termine dell'esecuzione vengono prodotte n righe di output della forma

        d i j dist

che mostrano la distanza dist tra il nodo $i$ e il nodo $j$ ($i$ sarà
sempre il nodo sorgente). Le caratteristiche dei grafi, e le distanze
tra il nodo $0$ e il nodo $n-1$ sono indicati nella tabella seguente:

Grafo                       Nodi (n)    Archi (m)    Distanza $0 \rightarrow n-1$
-------------------------  --------- ------------ -------------------------------
[rome99.gr](rome99.g)           3353         8870                         30290.0
[DE.gr](DE.gr)                 49109       121024                         69204.0
[VT.gr](VT.gr)                 97975       215116                        129866.0
[ME.gr](ME.gr)                194505       429842                        108545.0
[NV.gr](NV.gr)                261155       622086                        188894.0

Scopo dell'esercizio è di parallelizzare la funzione
`floyd_warshall()`.

Alcuni suggerimenti implementativi. Dato che le distanze sono di tipo
`double`, è possibile rappresentare il valore $\infty$ usando il
simbolo `INFINITY` definito in `math.h`. Tale valore si comporta
essenzialmente come il valore $\infty$, nel senso che risulta sempre
maggiore di qualsiasi valore finito di tipo `double`. Per controllare
se una variabile `x` ha valore infinito si usi la funzione `isinf(x)`,
che ritorna _true_ (nonzero) se e solo se `x` ha valore `INFINITY`.

## Files

- [omp-floyd-warshall.c](omp-floyd-warshall.c)
- Grafi di esempio: [rome99.gr](rome99.gr), [DE.gr](DE.gr), [VT.gr](VT.gr), [ME.gr](ME.gr), [NV.gr](NV.gr)

***/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> /* for isinf(), fminf() and HUGE_VAL */
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#else
double omp_get_wtime( void ) { return 0.0; }
#endif

typedef struct {
    int src, dst;
    double w;
} edge_t;

typedef struct {
    int n; /* number of nodes */
    int m; /* length of the edges array */
    edge_t *edges; /* array of edges */
} graph_t;

/**
 * Load a graph description in DIMACS format from file `f`; store the
 * graph in `g` (the caller is responsible for providing a pointer to
 * an already allocated object). This function interprets `g` as a
 * directed graph. For more information on the DIMACS challenge format
 * see http://www.diag.uniroma1.it/challenge9/index.shtml
 */
void load_dimacs(FILE *f, graph_t* g)
{
    const size_t buflen = 1024;
    char buf[buflen], prob[buflen];
    int n, m, src, dst, w;
    int idx = 0; /* index in the edge array */
    int nmatch;

    while ( fgets(buf, buflen, f) ) {
        switch( buf[0] ) {
        case 'c':
            break; /* ignore comment lines */
        case 'p':
            /* Parse problem format; expect type "shortest path" */
            sscanf(buf, "%*c %s %d %d", prob, &n, &m);
            if (strcmp(prob, "sp")) {
                fprintf(stderr, "FATAL: unknown DIMACS problem type %s\n", prob);
                exit(-1);
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
            fprintf(stderr, "FATAL: unrecognized character %c in line \"%s\"\n", buf[0], buf);
            exit(EXIT_FAILURE);
        }
    }
    assert( idx == g->m );
    /* sort_edges(g);  */
}

int IDX(int i, int j, int width)
{
    assert(i<width);
    assert(j<width);
    return i * width + j;
}

int floyd_warshall( const graph_t *g, double *d, int *p )
{
    assert(g != NULL);

    const int n = g->n;
    const int m = g->m;

#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (int u=0; u<n; u++) {
            for (int v=0; v<n; v++) {
                d[IDX(u,v,n)] = (u == v ? 0.0 : HUGE_VAL);
                p[IDX(u,v,n)] = -1;
            }
        }

#pragma omp for
        for (const edge_t *e = g->edges; e < g->edges + m; e++) {
            d[IDX(e->src,e->dst,n)] = e->w;
            p[IDX(e->src,e->dst,n)] = e->src;
        }

        for (int k=0; k<n; k++) {
#pragma omp for collapse(2)
            for (int u=0; u<n; u++) {
                for (int v=0; v<n; v++) {
                    if (d[IDX(u,k,n)] + d[IDX(k,v,n)] < d[IDX(u,v,n)]) {
                        d[IDX(u,v,n)] = d[IDX(u,k,n)] + d[IDX(k,v,n)];
                        p[IDX(u,v,n)] = p[IDX(k,v,n)];
                    }
                }
            }
        }
    } // pragma omp parallel

    /* Check for self-loops of negative cost. */
    for (int u=0; u<n; u++) {
        if ( d[IDX(u,u,n)] < 0.0 ) {
            /* printf("d[%d][%d] = %f\n", u, u, d[u][u]); */
            return 1;
        }
    }

    return 0;
}

int main( int argc, char* argv[] )
{
    graph_t g;
    double *d;
    int *p;

    if ( argc > 1 ) {
        fprintf(stderr, "Usage: %s < problem_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    load_dimacs(stdin, &g);

    /* Care must be taken to convert g.n to (size_t) in order to avoid
       overflows is the number of nodes is very large */
    d = (double*)malloc((size_t)g.n * (size_t)g.n * sizeof(*d)); assert(d);
    p = (int*)malloc((size_t)g.n * (size_t)g.n * sizeof(*p)); assert(p);

    const float tstart = omp_get_wtime();
    floyd_warshall(&g, d, p);
    const float elapsed = omp_get_wtime() - tstart;
    fprintf(stderr, "Execution time....... %f\n", elapsed);

    printf("d[%d,%d] = %f\n", 0, g.n-1, d[IDX(0, g.n-1, g.n)]);
    free(d);
    free(p);
    free(g.edges);
    return 0;
}
