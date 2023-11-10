/******************************************************************************
 *
 * omp-bellman-ford.c - Single-source shortest paths
 *
 * Copytight (C) 2017, 2018, 2023 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Single-Source Shortest Path
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-03-27

Lo scopo di questo esercizio è sviluppare una versione parallela
dell'algoritmo di Dijkstra per il calcolo dei cammini minimi da
singola sorgente (_Single-Source Shortest Paths_). Dato un grafo
orientato pesato con $n$ nodi e $m$ archi, i cui archi hanno pesi non
negativi, l'algoritmo di Dijkstra calcola la distanza (lunghezza del
cammino di costo minimo) $d[j]$ di ciascun nodo $j$, $0 \leq j < n$ da
un nodo sorgente $s$ (la distanza del nodo sorgente da se stesso è
zero, quindi $d[s] = 0$). L'uso di una coda di priorità consente di
realizzare l'algoritmo in modo che il costo asintotico sia $O(m + n
\log n)$; in questo esercizio però useremo una implementazione più
semplice e molto meno efficiente, che ha però il vantaggio di poter
essere parallelizzata in modo semplice.

L'algoritmo di Dijkstra costruisce l'albero dei cammini minimi in modo
iterativo, aggiungendo un nuovo arco ad ogni iterazione fino a
raggiungere tutti i nodi collegati con la sorgente. Detta $d[i]$ la
distanza del nodo $i$ dalla sorgente $s$. Inizialmente tutte le
distanze sono poste a $\infty$. Ad ogni iterazione si estende l'albero
dei cammini minimi aggiungendo l'arco $(i, j)$, tale che:

1. il nodo $i$ sia già stato raggiunto dall'albero dei cammini minimi (cioè $d[i] < \infty$);
2. il nodo $j$ non sia ancora stato raggiunto dall'albero dei cammini minimi (cioè $d[j] = \infty$)
3. la quantità $d[i] + w(i, j)$ sia minima tra tutti gli archi che soddisfano le due proprietà precedenti.

La scelta dell'arco da aggiungere ad ogni passo viene fatta esaminando
tutti gli archi del grafo (ciò rende l'implementazione proposta
inefficiente). Lo pseudocodice dell'algoritmo può essere descritto
come segue:

```
Dijkstra(grafo G=(V, E, w), int s)
	double d[0..n - 1]
	// Inizializza tutte le distanze a +∞
	for i ← 0 to n – 1 do
		d[i] ← +∞
	endfor
	d[s] ← 0
	do
		best_dist ← +∞ // minima distanza dalla sorgente tra tutti i nodi non ancora raggiunti
		best_node ← -1 // nodo non ancora raggiunto con minima distanza dalla sorgente
		foreach edge (i, j) Î E do
			if (d[i] < +∞ and d[j] = +∞ and d[i] + w(i, j) < best_dist) then
				best_dist ← d[i] + w(i, j)
				best_node ← j
			endif
		endfor
		if ( best_node ≠ -1 ) then
			d[best_node] ← best_dist
		endif
	while ( best_node ≠ -1 ) // Fino a quando abbiamo raggiunto tutti i nodi
```

Il file [omp-bellman-ford.c](omp-bellman-ford.c) contiene
l'implementazione seriale dell'algoritmo precedente. Il programma
accetta un parametro intero opzionale sulla riga di comando, che
rappresenta l'id del nodo sorgente (default 0). Il programma legge da
standard input una descrizione del grafo in formato DIMACS. Vengono
forniti alcuni esempi: [rome99.gr](rome99.gr) (porzione di mappa
stradale di Roma), [DE.gr](DE.gr) (porzione di mappa del Delaware),
[VT.gr](VT.gr) (porzione di mappa del Vermont), [ME.gr](ME.gr)
(porzione di mappa del Maine) e [NV.gr](NV.gr) (porzione di mappa del
Nevada). Suggerisco di iniziare con [rome99.gr](rome99.gr) perché
rappresenta il grafo più piccolo. L'elaborazione del grafo del Nevada
richiede molto tempo. Al termine dell'esecuzione vengono prodotte n
righe di output della forma

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

È possibile stampare la distanza tra la sorgente e il nodo $n-1$ con
il comando:

        ./omp-bellman-ford < rome99.gr | tail -1

Si presti attenzione al fatto che i grafi nei file di input sono da
considerarsi come grafi non orientati; questo significa che ogni arco
$(u, v)$ ompare due volte nell'array `edges[]` della struttura `graph_t`:
una volta come $(u, v)$ e una come $(v, u)$. Il campo `m` della struttura
`graph_t` indica la lunghezza dell'array degli archi, che quindi è il
doppio del numero di archi che compaiono nel file di input.

Scopo dell'esercizio è di parallelizzare la funzione
`dijkstra()`. Analizzando il codice della funzione si vede che l'unico
punto su cui è facile intervenire è il ciclo "for" che esplora la
lista degli archi per determinare l'arco (se esiste) che conduce al
nodo non ancora esplorato avente distanza minima dalla sorgente. Nel
programma la lista di archi è rappresentata mediante un array, per cui
è possibile dividere l'array in blocchi e assegnare ogni blocco ad un
thread OpenMP. Per fare ciò non è però conveniente utilizzare un
costrutto `omp parallel for`: infatti, al termine del ciclo dobbiamo
determinare non solo la minima distanza tra i nodi non ancora
raggiunti (`best_dist`), ma anche l'id del nodo a distanza minima
(`best_node`). La minima distanza può essere calcolata usando una
clausola `reduction`; purtroppo non è altrettanto facile determinare
anche l'id del nodo a distanza minima. Di conseguenza dobbiamo
effettuare una parallelizzazione "manuale" come quella che abbiamo
visto nei primi esercizi usando un costrutto `omp parallel`. In
particolare:

- Ciascun thread $t$ determina gli estremi del sottovettore dell'array
  di archi di sua competenza;

- Ogni thread $t$ esamina gli archi nel sottovettore di cui al punto
  precedente, determinando la minima distanza `my_best_dist` tra i
  nodi non ancora esplorati che possono essere raggiunti tramite tali
  archi, e l'id `my_best_node` del nodo avente tale distanza minima;

- Tra tutte le coppie (`my_best_dist`, `my_best_node`) si sceglie quella
  il cui valore `my_best_dist` è minimo.

Ci sono diversi modi con cui è possibile realizzare quanto sopra. Ad
esempio, si potrebbero condividere due array condivisi
`my_best_dist[]` e `my_best_node[]` i cui elementi rappresentano i
valori della distanza minima e del nodo con distanza minima calcolati
dal thread $t$-esimo; al termine della fase parallela il master
determina la coppia da usare scorrendo gli array. alternativamente, al
termine della fase parallela ciascun thread confronta, in una sezione
critica, il proprio valore `my_best_dist` con il minimo globale
`best_dist`, e se quest'ultimo risulta superiore del minimo locale si
aggiorna.

Alcuni suggerimenti implementativi. Dato che le distanze sono di tipo
`double`, è possibile rappresentare il valore $\infty$ usando il
simbolo `INFINITY` definito in `math.h`. Tale valore si comporta
essenzialmente come il valore $\infty$, nel senso che risulta sempre
maggiore di qualsiasi valore finito di tipo `double`. Per controllare
se una variabile `x` ha valore infinito si usi la funzione `isinf(x)`,
che ritorna _true_ (nonzero) se e solo se `x` ha valore `INFINITY`.

## Files

- [omp-bellman-ford.c](omp-bellman-ford.c)
- Grafi di esempio: [rome99.gr](rome99.gr), [DE.gr](DE.gr), [VT.gr](VT.gr), [ME.gr](ME.gr), [NV.gr](NV.gr)

***/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> /* for isinf(), fminf() and HUGE_VAL */
#include <assert.h>
#include <omp.h>

typedef struct {
    int src, dst;
    float w;
} edge_t;

typedef struct {
    int n; /* number of nodes */
    int m; /* length of the edges array */
    edge_t *edges; /* array of edges */
} graph_t;

int cmp_edges(const void* p1, const void* p2)
{
    edge_t *e1 = (edge_t*)p1;
    edge_t *e2 = (edge_t*)p2;
    return (e1->dst - e2->dst);
}

void sort_edges(graph_t *g)
{
    qsort(g->edges, g->m, sizeof(edge_t), cmp_edges);
}

/* Set *v = min(*v, x), atomically; return 1 iff the value of *v changed */
static inline int atomicRelax(volatile float *v, float x)
{
    union {
        float vf;
        int vi;
    } oldval, newval;

    if ( *v <= x )
        return 0;

    do {
        oldval.vf = *v;
        newval.vf = fminf(oldval.vf, x);
    } while ( (oldval.vf > newval.vf) && ! __sync_bool_compare_and_swap((int*)v, oldval.vi, newval.vi) );
    return (newval.vf != oldval.vf);
}

/**
 * Load a graph description in DIMACS format from file |f|; store the
 * graph in |g| (the caller is responsible for providing a pointer to
 * an already allocated object). This function interprets |g| as an
 * _undirected_ graph; this means that each edge (u,v) in the input
 * file appears twice in |g|, as (u,v) and (v,u)
 * respectively. Therefore, if the input graph has |m| edges, the edge
 * array of |g| will have 2*m elements. For more information on the
 * DIMACS challenge format see
 * http://www.diag.uniroma1.it/challenge9/index.shtml
 */
void load_dimacs(FILE *f, graph_t* g)
{
    const size_t buflen = 1024;
    char buf[buflen], prob[buflen];
    int n, m, src, dst, w;
    int cnt = 0; /* edge counter */
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
            g->m = 2*m;
            g->edges = (edge_t*)malloc((g->m)*sizeof(edge_t)); assert(g->edges);
            cnt = idx = 0;
            break;
        case 'a':
            nmatch = sscanf(buf, "%*c %d %d %d", &src, &dst, &w);
            if (nmatch != 3) {
                fprintf(stderr, "FATAL: Malformed line \"%s\"\n", buf);
                exit(-1);
            }
            /* In the DIMACS format, nodes are numbered starting from
               1; we use zero-based indexing internally, so we
               decrement the ids by one */

            /* For each edge (u,v,w) in the input file, we insert two
               edges (u,v,w) and (v,u,w) in the edge array, one edge
               for each direction */
            g->edges[idx].src = src-1;
            g->edges[idx].dst = dst-1;
            g->edges[idx].w = w;
            idx++;
            g->edges[idx].src = dst-1;
            g->edges[idx].dst = src-1;
            g->edges[idx].w = w;
            idx++;
            cnt++;
            break;
        default:
            fprintf(stderr, "FATAL: unrecognized character %c in line \"%s\"\n", buf[0], buf);
            exit(-1);
        }
    }
    assert( 2*cnt == g->m );
    /* sort_edges(g);  */
}

/* Compute distances from source node |s| using Dijkstra's algorithm.
   This implementation is extremely inefficient since it traverses the
   whole edge list at each iteration, while a smarter implementation
   would traverse only the edges incident to the frontier of the
   shortest path tree. However, it is quite easy to parallelize, and
   this is what counts here. |g| is the input graph; |s| is the source
   node id; |d| is the array of distances, that must be pre-allocated
   by the caller to hold |g->n| elements. When the function
   terminates, d[i] is the distance of node i from node s. */
void dijkstra(const graph_t* g, int s, float *d)
{
    const int n = g->n;
    const int m = g->m;
    float best_dist; /* minimum distance */
    int best_node; /* node with minimum distance */
    int i;

    for (i=0; i<n; i++) {
        d[i] = INFINITY;
    }
    d[s] = 0;
    do {
        best_dist = INFINITY;
        best_node = -1;
#pragma omp parallel default(none) shared(g,d,best_dist,best_node,m)
        {
            float my_best_dist = INFINITY;
            int my_best_node = -1;
            int j;

#pragma omp for
            for (j=0; j<m; j++) {
                const int src = g->edges[j].src;
                const int dst = g->edges[j].dst;
                const float dist = d[src] + g->edges[j].w;
                if ( isfinite(d[src]) && isinf(d[dst]) && (dist < my_best_dist) ) {
                    my_best_dist = dist;
                    my_best_node = dst;
                }
            }
#pragma omp critical
            {
                if ( my_best_dist < best_dist ) {
                    best_dist = my_best_dist;
                    best_node = my_best_node;
                }
            }
        }
        if ( isfinite(best_dist) ) {
            assert( best_node >= 0 );
            d[best_node] = best_dist;
        }
    } while (isfinite(best_dist));
}

/* Compute shortest paths from node s using Bellman-Ford
   algorithm. This is a simple serial algorithm (probably the simplest
   algorithm to compute shortest paths), that might be used to check
   the correctness of function dijkstra(). This version is slightly
   optimized with respect to the "canonical" implementation of the
   Bellman-Ford algorithm: if no distance is updated after a
   relaxation phase, this function terminates immediately since no
   distances will be updated in future iterations. */
void bellmanford(const graph_t* g, int s, float *d)
{
    const int n = g->n;
    const int m = g->m;
    int i, j, updated, niter = 0;

    for (i=0; i<n; i++) {
        d[i] = INFINITY;
    }
    d[s] = 0;
    do {
        updated = 0;
        niter++;
        for (j=0; j<m; j++) {
            const int src = g->edges[j].src;
            const int dst = g->edges[j].dst;
            const float w = g->edges[j].w;
            if ( d[src] + w < d[dst] ) {
                d[dst] = d[src] + w;
                updated = 1;
            }
        }
    } while (updated);
    fprintf(stderr, "bellmanford: %d iterations\n", niter);
}

/* Using atomic to protect updates to the new distances */
void bellmanford_atomic(const graph_t* g, int s, float *d)
{
    const int n = g->n;
    const int m = g->m;
    int i, j, updated, niter = 0;

#pragma omp parallel for default(none) shared(d,n)
    for (i=0; i<n; i++) {
        d[i] = INFINITY;
    }
    d[s] = 0.0f;
    do {
        updated = 0;
        niter++;
#pragma omp parallel for default(none) shared(g, d, m) reduction(|:updated)
        for (j=0; j<m; j++) {
            const int src = g->edges[j].src;
            const int dst = g->edges[j].dst;
            const float w = g->edges[j].w;
            if (d[src]+w < d[dst]) {
                updated |= atomicRelax(&d[dst], d[src]+w);
            }
        }
    } while (updated);
    fprintf(stderr, "bellmanford_atomic: %d iterations\n", niter);
}

void bellmanford_atomic_inlined(const graph_t* g, int s, float *d)
{
    const int n = g->n;
    const int m = g->m;
    int i, j, updated, niter = 0;

#pragma omp parallel for
    for (i=0; i<n; i++) {
        d[i] = INFINITY;
    }
    d[s] = 0.0f;
    do {
        updated = 0;
        niter++;
#pragma omp parallel for default(none) shared(g, d, m) reduction(|:updated)
        for (j=0; j<m; j++) {
            const int src = g->edges[j].src;
            const int dst = g->edges[j].dst;
            const float w = g->edges[j].w;

            if ( d[src] + w < d[dst] ) {
                union {
                    float vf;
                    int vi;
                } oldval, newval;

                volatile int* dist_p = (int*)(d + dst);

                do {
                    oldval.vf = d[dst];
                    newval.vf = fminf(d[src]+w, d[dst]);
                } while ((oldval.vf > newval.vf) && ! __sync_bool_compare_and_swap(dist_p, oldval.vi, newval.vi) );
                updated |= (newval.vi != oldval.vi);
            }
        }
    } while (updated);
    fprintf(stderr, "bellmanford_atomic: %d iterations\n", niter);
}

/* Bellman-Ford algorithm without syncronization. Note that this
   implementation is technically NOT CORRECT, since multiple OpenMP
   threads might "relax" the same distance at the same time, resulting
   in a race condition. However, for some reasons that I do not
   understand, the program seems to always compute the correct
   distance on the test cases I tried. */
void bellmanford_none(const graph_t* g, int s, float *d)
{
    const int n = g->n;
    const int m = g->m;
    int i, j, updated = 0, niter = 0;

#pragma omp parallel for
    for (i=0; i<n; i++) {
        d[i] = INFINITY;
    }
    d[s] = 0.0f;
    do {
        updated = 0;
        niter++;
#pragma omp parallel for default(none) shared(g, d, m) reduction(|:updated)
        for (j=0; j<m; j++) {
            const int src = g->edges[j].src;
            const int dst = g->edges[j].dst;
            const float w = g->edges[j].w;
            if ( d[src]+w < d[dst] ) {
                updated = 1;
                d[dst] = d[src]+w;
            }
        }
    } while (updated);
    fprintf(stderr, "belmanford_none: %d iterations\n", niter);
}

/* Check distances. Return 0 iff d1 and d2 contain the same values (up
   to a given tolerance), -1 otherwise. */
int checkdist( float *d1, float *d2, int n)
{
    const float TOL = 1e-5;
    int i;
    for (i=0; i<n; i++) {
        if ( fabsf(d1[i] - d2[i]) > TOL ) {
            fprintf(stderr, "FATAL: d1[%d]=%f, d2[%d]=%f\n", i, d1[i], i, d2[i]);
            return -1;
        }
    }
    return 0;
}

int main( int argc, char* argv[] )
{
    graph_t g;
    int src = 0;
    float *d_serial, *d_parallel;
    float tstart, t_serial, t_atomic, t_none, t_dijkstra;

    /* required by atomicRelax() */
    assert( sizeof(float) == sizeof(int) );

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [source_node] < problem_file > distance_file\n", argv[0]);
        return -1;
    }

    load_dimacs(stdin, &g);

    const size_t sz = (g.n) * sizeof(*d_serial);

    d_serial = (float*)malloc(sz); assert(d_serial);
    d_parallel = (float*)malloc(sz); assert(d_parallel);

    if ( argc > 1 ) {
        src = atoi(argv[1]);
        if (src < 0 || src >= g.n) {
            fprintf(stderr, "FATAL: invalid source node (should be within %d-%d)\n", 0, g.n-1);
            exit(-1);
        }
    }

    tstart = omp_get_wtime();
    bellmanford(&g, src, d_serial);
    t_serial = omp_get_wtime() - tstart;
    fprintf(stderr, "Serial execution time....... %f\n", t_serial);

    tstart = omp_get_wtime();
    bellmanford_atomic(&g, src, d_parallel);
    t_atomic = omp_get_wtime() - tstart;
    fprintf(stderr, "Bellman-Ford (atomic)....... %f (%.2fx)\n", t_atomic, t_serial/t_atomic);
    checkdist(d_serial, d_parallel, g.n);

    tstart = omp_get_wtime();
    bellmanford_none(&g, src, d_parallel);
    t_none = omp_get_wtime() - tstart;
    fprintf(stderr, "Bellman-Ford (no sync)...... %f (%.2fx)\n", t_none, t_serial/t_none);
    checkdist(d_serial, d_parallel, g.n);

    tstart = omp_get_wtime();
    dijkstra(&g, src, d_parallel);
    t_dijkstra = omp_get_wtime() - tstart;
    fprintf(stderr, "Dijkstra.................... %f (%.2fx)\n", t_dijkstra, t_serial/t_dijkstra);
    checkdist(d_serial, d_parallel, g.n);

    /* print distances to stdout */
#if 0
    for (i=0; i<g.n; i++) {
        printf("d %d %d %f\n", src, i, d_serial[i]);
    }
#endif

    free(d_serial);
    free(d_parallel);
    return 0;
}
