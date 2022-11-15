/****************************************************************************
 *
 * cuda-anneal.cu - ANNEAL cellular automaton with CUDA
 *
 * Copyright (C) 2017--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - L'automa cellulare ANNEAL
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2021-11-28

In questo esercizio consideriamo un semplice automa cellulare binario
in due dimensioni, denominato _ANNEAL_ (noto anche come _twisted
majority rule_). L'automa opera su un dominio quadrato di dimensione
$N \times N$, in cui ogni cella può avere valore 0 oppure 1. Si assume
un dominio toroidale, in modo che ogni cella, incluse quelle sul
bordo, abbia sempre otto celle adiacenti. Due celle si considerano
adiacenti se hanno un lato oppure uno spigolo in comune.

L'automa evolve a istanti di tempo discreti $t = 0, 1, 2, \ldots$. Lo
stato di una cella al tempo $t+1$ dipende dal proprio stato e da
quello degli otto vicini al tempo $t$. In dettaglio, per ogni cella
$x$ sia $B_x$ il numero di celle con valore 1 presenti nell'intorno di
dimensione $3 \times 3$ centrato su $x$ (si conta anche lo stato di
$x$, quindi si avrà sempre $0 \leq B_x \leq 9$). Se $B_x=4$ oppure
$B_x \geq 6$, allora il nuovo stato della cella $x$ è 1; in caso
contrario il nuovo stato è 0. La Figura 1 mostra alcuni esempi.

![Figura 1: Esempi di calcolo del nuovo stato della cella centrale di un blocco di dimensione $3 \times 3$](cuda-anneal1.png)

Come sempre in questi casi si devono utilizzare due griglie (domini)
per rappresentare lo stato corrente dell'automa e lo stato al passo
successivo. Lo stato delle celle viene sempre letto dalla griglia
corrente, e i nuovi valori vengono sempre scritti nella griglia
successiva. Quando il nuovo stato di tutte le celle è stato calcolato,
si scambiano le griglie e si ripete.

Il dominio viene inizializzato ponendo ogni cella a 0 o 1 con uguale
probabilità; di conseguenza, circa metà delle celle saranno nello
stato 0 e l'altra metà sarà nello stato 1. La Figura 2 mostra
l'evoluzione di una griglia di dimensione $256 \times 256$ dopo 10,
100 e 1024 iterazioni. Si può osservare come le celle 0 e 1 tendano
progressivamente ad addensarsi, pur con la presenza di piccole
"bolle". È disponibile un [breve video che mostra l'evoluzione
dell'automa](https://youtu.be/UNpl2iUyz3Q) nel tempo.

![Figura 2: Evoluzione dell'automa _ANNEAL_ ([video](https://youtu.be/UNpl2iUyz3Q))](anneal-demo.png)

Il file [cuda-anneal.cu](cuda-anneal.cu) contiene una implementazione
seriale dell'algoritmo che calcola e salva su un file l'evoluzione
dopo $K$ iterazioni dell'automa cellulare basato sulla regola
ANNEAL. Scopo di questo esercizio è di modificare il programma per
delegare alla GPU sia il calcolo del nuovo stato, sia la copia dei
bordi del dominio (necessaria per simulare un dominio toroidale).

Alcuni suggerimenti:

- Iniziare sviluppando una versione che non usa la memoria
  `__shared__`. Trasformare le funzioni `copy_top_bottom()`,
  `copy_left_right()` e `step()` in kernel; in questo modo è possibile
  fare evolvere l'automa interamente nella memoria della GPU. La
  dimensione dei thread block necessari a copiare le celle sarà
  diverso dalla dimensione dei blocchi usati per l'evoluzione
  dell'automa (vedi punti seguenti).

- Per copiare le ghost cell ai lati è sufficiente organizzare i thread
  in un array (blocchi 1D). Quindi per l'esecuzione dei kernel
  `copy_top_bottom()` e `copy_left_right()` saranno necessari $(N+2)$
  thread.

- Dato che il dominio è bidimensionale, per calcolare l'evoluzione
  dell'automa conviene usare blocchi bidimensionali di thread. Usando
  blocchi di thread di dimensioni $\mathit{BLKDIM} \times
  \mathit{BLKDIM}$, la griglia dovrà avere dimensioni $(N +
  \mathit{BLKDIM} - 1)/\mathit{BLKDIM} \times (N + \mathit{BLKDIM} -
  1)/\mathit{BLKDIM}$. Ricordarsi che la GPU consente al massimo 1024
  thread per blocco; suggerisco quindi di usare $\mathit{BLKDIM}=32$
  in modo che un blocco sia composto esattamente da 1024 CUDA thread.

- Nel kernel `step()`, ciascun thread calcola il nuovo stato di una
  cella di coordinate $(i, j)$. Ricordare che si sta lavorando su un
  dominio "allargato" con due righe e due colonne in più, quindi le
  celle "vere" (non ghost) sono quelle con coordinate $1 \leq i, j
  \leq N$. Di conseguenza, ogni thread calcolerà $i, j$ come:
```C
  const int i = 1 + threadIdx.y + blockIdx.y * blockDim.y;
  const int j = 1 + threadIdx.x + blockIdx.x * blockDim.x;
```
  In questo modo i thread verranno associati alle celle di coordinate
  da $(1, 1)$ in poi. Prima di effettuare qualsiasi computazione,
  ogni thread dovrà verificare che $1 \leq i, j \leq N$, in modo
  tale che eventuali thread in eccesso vengano disattivati.

## Uso della _shared memory_

Questo programma potrebbe beneficiare dall'uso della memoria
`__shared__`, dato che ogni cella del dominio viene letta 9 volte da 9
thread diversi. Tuttavia, **sul server non si noterà alcun
miglioramento delle prestazioni** perché le GPU sono dotate di memoria
_cache_ e il numero di riletture non è abbastanza elevato da
ammortizzare il costo della copia verso la memoria
condivisa. Nonostante questo, è un esercizio istruttivo realizzare una
versione che sfrutti la memoria _shared_.

Assumiamo che $N$ sia un multiplo esatto di _BLKDIM_. Ciascun blocco di
thread copia gli elementi della porzione di dominio di sua competenza
in un buffer locale `buf[BLKDIM+2][BLKDIM+2]` che include due righe e
due colonne (le prime e le ultime) di ghost cell, e calcolare il
nuovo stato delle celle usando i dati nel buffer locale anziché
accedendo alla memoria globale.

In situazioni del genere è utile usare due coppie di indici $(gi, gj)$
per indicare le posizioni delle celle nella matrice globale e $(li,
lj)$ per le posizioni delle celle nel buffer locale. L'idea è che la
cella di coordinate $(gi, gj)$ nella matrice globale corrisponda a
quella di coordinate $(li, lj)$ nel buffer locale. Usando ghost cell
sia a livello globale che a livello locale il calcolo delle coordinate
può essere effettuato come segue:

```C
    const int gi = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    const int gj = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    const int li = 1 + threadIdx.y;
    const int lj = 1 + threadIdx.x;
```

![Figura 3: Copia dei dati dal dominio globale verso la shared memory](cuda-anneal3.png)

La parte più laboriosa è la copia dei dati dalla griglia globale al
buffer locale. Usando $\mathit{BLKDIM} \times \mathit{BLKDIM}$ thread
per blocco, la copia della parte centrale (cioè tutto ad esclusione
dell'area tratteggiata della Figura 3) si effettua con:

```C
    buf[li][lj] = *IDX(cur, ext_n, gi, gj);
```

dove `ext_n = (N + 2)` è il lato del dominio inclusa la ghost
area.

![Figura 4: Thread attivi durante il riempimento del dominio locale](cuda-anneal4.png)

Per inizializzare la ghost area occorre procedere in tre fasi (Figura 4):

1. La ghost area superiore e inferiore viene delegata ai thread della
   prima riga (quelli con $li = 1$);

2. La ghost area a sinistra e a destra viene delegata ai thread della
   prima colonna (quelli con $lj = 1$);

3. La ghost area negli angoli viene delegata al singolo thread con
   $(li, lj) = (1, 1)$.

(Si potrebbe essere tentati di collassare le fasi 1 e 2 in un'unica
fase da far svolgere ad esempio ai thread della prima riga; questo
sarebbe corretto, ma presenterebbe problemi nel caso si decidesse di
generalizzare il codice a lati del dominio non necessariamente
multipli di $\mathit{BLKDIM}$).

Si avrà in pratica la struttura seguente:

```C
    if ( li == 1 ) {
        "riempi la cella buf[0][lj] e buf[BLKDIM+1][lj]"
    }
    if ( lj == 1 ) {
        "riempi la cella buf[li][0] e buf[li][BLKDIM+1]"
    }
    if ( li == 1 && lj == 1 ) {
        "riempi buf[0][0]"
        "riempi buf[0][BLKDIM+1]"
        "riempi buf[BLKDIM+1][0]"
        "riempi buf[BLKDIM+1][BLKDIM+1]"
    }
```

Chi vuole cimentarsi con una versione ancora più laboriosa può provare
a modificare il codice per gestire anche il caso in cui la dimensione
del dominio non sia multipla di _BLKDIM_. Prestare attenzione che non
è sufficiente disattivare i thread al di fuori del dominio, ma bisogna
modificare l'operazione di la copia della ghost area.

Per compilare senza usare shared memory:

        nvcc cuda-anneal.cu -o cuda-anneal

Per generare una immagine ad ogni passo:

        nvcc -DDUMPALL cuda-anneal.cu -o cuda-anneal

È possibile montare le immagini in un video in formato AVI/MPEG-4 con:

        ffmpeg -y -i "cuda-anneal-%06d.pbm" -vcodec mpeg4 cuda-anneal.avi

(il comando `ffmpeg` è già installato sul server).

Per compilare la soluzione abilitando la shared memory:

        nvcc -DUSE_SHARED cuda-anneal.cu -o cuda-anneal-shared

Per eseguire:

        ./cuda-anneal [steps [N]]

Esempio:

        ./cuda-anneal 64

## File

- [cuda-anneal.cu](cuda-anneal.cu)
- [Animazione](https://youtu.be/UNpl2iUyz3Q)
- [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifndef SERIAL
/* We use 2D blocks of size (BLKDIM * BLKDIM) to compute
   the next configuration of the automaton */

#define BLKDIM 32

/* We use 1D blocks of (BLKDIM_COPY) threads to copy ghost cells */

#define BLKDIM_COPY 1024
#endif

typedef unsigned char cell_t;

/* The following function makes indexing of the 2D domain
   easier. Instead of writing, e.g., grid[i*ext_n + j] you write
   IDX(grid, ext_n, i, j) to get a pointer to grid[i][j]. This
   function assumes that the size of the CA grid is (ext_n*ext_n),
   where the first and last rows/columns are ghost cells. */
#ifndef SERIAL
__device__ __host__
#endif
cell_t* IDX(cell_t *grid, int ext_n, int i, int j)
{
    return (grid + i*ext_n + j);
}

#ifndef SERIAL
__host__ __device__
#endif
int d_min(int a, int b)
{
    return (a<b ? a : b);
}

/*
  `grid` points to a (ext_n * ext_n) block of bytes; this function
  copies the top and bottom ext_n elements to the opposite halo (see
  figure below).

   LEFT_GHOST=0     RIGHT=ext_n-2
   | LEFT=1         | RIGHT_GHOST=ext_n-1
   | |              | |
   v v              v v
  +-+----------------+-+
  |Y|YYYYYYYYYYYYYYYY|Y| <- TOP_GHOST=0
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- TOP=1
  | |                | |
  | |                | |
  | |                | |
  | |                | |
  |Y|YYYYYYYYYYYYYYYY|Y| <- BOTTOM=ext_n - 2
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- BOTTOM_GHOST=ext_n - 1
  +-+----------------+-+

 */
#ifdef SERIAL
/* [TODO] Transform this function into a kernel */
void copy_top_bottom(cell_t *grid, int ext_n)
{
    int j;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;

    for (j=0; j<ext_n; j++) {
        *IDX(grid, ext_n, BOTTOM_GHOST, j) = *IDX(grid, ext_n, TOP, j); /* top to bottom halo */
        *IDX(grid, ext_n, TOP_GHOST, j) = *IDX(grid, ext_n, BOTTOM, j); /* bottom to top halo */
    }
}
#else
__global__ void copy_top_bottom(cell_t *grid, int ext_n)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;

    if (j < ext_n) {
        *IDX(grid, ext_n, BOTTOM_GHOST, j) = *IDX(grid, ext_n, TOP, j); /* top to bottom halo */
        *IDX(grid, ext_n, TOP_GHOST, j) = *IDX(grid, ext_n, BOTTOM, j); /* bottom to top halo */
    }
}
#endif

/*
  `grid` points to a ext_n*ext_n block of bytes; this function copies
  the left and right ext_n elements to the opposite halo (see figure
  below).

   LEFT_GHOST=0     RIGHT=ext_n-2
   | LEFT=1         | RIGHT_GHOST=ext_n-1
   | |              | |
   v v              v v
  +-+----------------+-+
  |X|Y              X|Y| <- TOP_GHOST=0
  +-+----------------+-+
  |X|Y              X|Y| <- TOP=1
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y| <- BOTTOM=ext_n - 2
  +-+----------------+-+
  |X|Y              X|Y| <- BOTTOM_GHOST=ext_n - 1
  +-+----------------+-+

 */
#ifdef SERIAL
/* [TODO] This function should be transformed into a kernel */
void copy_left_right(cell_t *grid, int ext_n)
{
    int i;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

    for (i=0; i<ext_n; i++) {
        *IDX(grid, ext_n, i, RIGHT_GHOST) = *IDX(grid, ext_n, i, LEFT); /* left column to right halo */
        *IDX(grid, ext_n, i, LEFT_GHOST) = *IDX(grid, ext_n, i, RIGHT); /* right column to left halo */
    }
}
#else
__global__ void copy_left_right(cell_t *grid, int ext_n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

    if (i < ext_n) {
        *IDX(grid, ext_n, i, RIGHT_GHOST) = *IDX(grid, ext_n, i, LEFT); /* left column to right halo */
        *IDX(grid, ext_n, i, LEFT_GHOST) = *IDX(grid, ext_n, i, RIGHT); /* right column to left halo */
    }
}
#endif

#ifdef SERIAL
/* Compute the `next` grid given the current configuration `cur`.
   Both grids have (ext_n*ext_n) elements.
   [TODO] This function should be transformed into a kernel. */
void step(cell_t *cur, cell_t *next, int ext_n)
{
    int i, j;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;
    for (i=TOP; i <= BOTTOM; i++) {
        for (j=LEFT; j <= RIGHT; j++) {
            const int nblack =
                *IDX(cur, ext_n, i-1, j-1) + *IDX(cur, ext_n, i-1, j) + *IDX(cur, ext_n, i-1, j+1) +
                *IDX(cur, ext_n, i  , j-1) + *IDX(cur, ext_n, i  , j) + *IDX(cur, ext_n, i  , j+1) +
                *IDX(cur, ext_n, i+1, j-1) + *IDX(cur, ext_n, i+1, j) + *IDX(cur, ext_n, i+1, j+1);
            *IDX(next, ext_n, i, j) = (nblack >= 6 || nblack == 4);
        }
    }
}
#else
/* Compute the next grid given the current configuration. Each grid
   has (ext_n*ext_n) elements. */
__global__ void step(cell_t *cur, cell_t *next, int ext_n)
{
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;
    const int i = TOP + threadIdx.y + blockIdx.y * blockDim.y;
    const int j = LEFT + threadIdx.x + blockIdx.x * blockDim.x;

    if ( i <= BOTTOM && j <= RIGHT ) {
        const int nblack =
            *IDX(cur, ext_n, i-1, j-1) + *IDX(cur, ext_n, i-1, j) + *IDX(cur, ext_n, i-1, j+1) +
            *IDX(cur, ext_n, i  , j-1) + *IDX(cur, ext_n, i  , j) + *IDX(cur, ext_n, i  , j+1) +
            *IDX(cur, ext_n, i+1, j-1) + *IDX(cur, ext_n, i+1, j) + *IDX(cur, ext_n, i+1, j+1);
        *IDX(next, ext_n, i, j) = (nblack >= 6 || nblack == 4);
    }
}

/* Same as above, but using shared memory. This kernel works correctly
   even if the size of the domain is not multiple of BLKDIM.

   Note that, on modern GPUs, this version is actually *slower* than
   the plain version above.  The reason is that neser GPUs have an
   internal cache, and this computation does not reuse data enough to
   pay for the cost of filling the shared memory. */
__global__ void step_shared(cell_t *cur, cell_t *next, int ext_n)
{
    __shared__ cell_t buf[BLKDIM+2][BLKDIM+2];

    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;

    /* "global" indexes */
    const int gi = TOP + threadIdx.y + blockIdx.y * blockDim.y;
    const int gj = LEFT + threadIdx.x + blockIdx.x * blockDim.x;
    /* "local" indexes */
    const int li = 1 + threadIdx.y;
    const int lj = 1 + threadIdx.x;

    /* The following variables are needed to handle the case of a
       domain whose size is not multiple of BLKDIM.

       height and width of the (NOT extended) subdomain handled by
       this thread block. Its maximum size is blockdim.x * blockDim.y,
       but could be less than that if the domain size is not a
       multiple of the block size. */
    const int height = d_min(blockDim.y, ext_n-1-gi);
    const int width  = d_min(blockDim.x, ext_n-1-gj);

    if ( gi <= BOTTOM && gj <= RIGHT ) {
        buf[li][lj] = *IDX(cur, ext_n, gi, gj);
        if (li == 1) {
            /* top and bottom */
            buf[0       ][lj] = *IDX(cur, ext_n, gi-1, gj);
            buf[1+height][lj] = *IDX(cur, ext_n, gi+height, gj);
        }
        if (lj == 1) { /* left and right */
            buf[li][0      ] = *IDX(cur, ext_n, gi, gj-1);
            buf[li][1+width] = *IDX(cur, ext_n, gi, gj+width);
        }
        if (li == 1 && lj == 1) { /* corners */
            buf[0       ][0       ] = *IDX(cur, ext_n, gi-1, gj-1);
            buf[0       ][lj+width] = *IDX(cur, ext_n, gi-1, gj+width);
            buf[1+height][0       ] = *IDX(cur, ext_n, gi+height, gj-1);
            buf[1+height][1+width ] = *IDX(cur, ext_n, gi+height, gj+width);
        }
        __syncthreads(); /* Wait for all threads to fill the shared memory */

        const int nblack =
            buf[li-1][lj-1] + buf[li-1][lj] + buf[li-1][lj+1] +
            buf[li  ][lj-1] + buf[li  ][lj] + buf[li  ][lj+1] +
            buf[li+1][lj-1] + buf[li+1][lj] + buf[li+1][lj+1];
        *IDX(next, ext_n, gi, gj) = (nblack >= 6 || nblack == 4);
    }
}
#endif

/* Initialize the current grid `cur` with alive cells with density
   `p`. */
void init( cell_t *cur, int ext_n, float p )
{
    int i, j;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;

    srand(1234); /* initialize PRND */
    for (i=TOP; i <= BOTTOM; i++) {
        for (j=LEFT; j <= RIGHT; j++) {
            *IDX(cur, ext_n, i, j) = (((float)rand())/RAND_MAX < p);
        }
    }
}

/* Write `cur` to a PBM (Portable Bitmap) file whose name is derived
   from the step number `stepno`. */
void write_pbm( cell_t *cur, int ext_n, int stepno )
{
    int i, j;
    char fname[128];
    FILE *f;

    snprintf(fname, sizeof(fname), "cuda-anneal-%06d.pbm", stepno);

    if ((f = fopen(fname, "w")) == NULL) {
        fprintf(stderr, "Cannot open %s for writing\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by cuda-anneal.cu\n");
    fprintf(f, "%d %d\n", ext_n-2, ext_n-2);
    for (i=1; i<ext_n-1; i++) {
        for (j=1; j<ext_n-1; j++) {
            fprintf(f, "%d ", *IDX(cur, ext_n, i, j));
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main( int argc, char* argv[] )
{
#ifdef SERIAL
    cell_t *cur, *next;
#else
    cell_t *cur;
    cell_t *d_cur, *d_next;
#endif
    int s, nsteps = 64, n = 512;
    const int MAXN = 2048;

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [nsteps [n]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        nsteps = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        n = atoi(argv[2]);
    }

    if ( n > MAXN ) { /* maximum image size is MAXN */
        fprintf(stderr, "FATAL: the maximum allowed grid size is %d\n", MAXN);
        return EXIT_FAILURE;
    }

    const int ext_n = n + 2;
    const size_t ext_size = ext_n * ext_n * sizeof(cell_t);

    fprintf(stderr, "Anneal CA: steps=%d size=%d\n", nsteps, n);
#ifndef SERIAL
#ifdef USE_SHARED
    printf("Using shared memory\n");
#else
    printf("NOT using shared memory\n");
#endif
#endif

#ifdef SERIAL
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    next = (cell_t*)malloc(ext_size); assert(next != NULL);
    init(cur, ext_n, 0.5);
    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        copy_top_bottom(cur, ext_n);
        copy_left_right(cur, ext_n);
#ifdef DUMPALL
        write_pbm(cur, ext_n, s);
#endif
        step(cur, next, ext_n);
        cell_t *tmp = cur;
        cur = next;
        next = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;
#else
    /* 1D blocks used for copying sides */
    const dim3 copyBlock(BLKDIM_COPY);
    const dim3 copyGrid((ext_n + BLKDIM_COPY-1)/BLKDIM_COPY);
    /* 2D blocks used for the update step */
    const dim3 stepBlock(BLKDIM, BLKDIM);
    const dim3 stepGrid((n + BLKDIM-1)/BLKDIM, (n + BLKDIM-1)/BLKDIM);

    /* Allocate space for host copy of the current grid */
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    /* Allocate space for device copy of |cur| and |next| grids */
    cudaSafeCall( cudaMalloc((void**)&d_cur, ext_size) );
    cudaSafeCall( cudaMalloc((void**)&d_next, ext_size) );

    init(cur, ext_n, 0.5);
    /* Copy initial grid to device */
    cudaSafeCall( cudaMemcpy(d_cur, cur, ext_size, cudaMemcpyHostToDevice) );

    /* evolve the CA */
    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        copy_top_bottom<<<copyGrid, copyBlock>>>(d_cur, ext_n); cudaCheckError();
        copy_left_right<<<copyGrid, copyBlock>>>(d_cur, ext_n); cudaCheckError();
#ifdef USE_SHARED
        step_shared<<<stepGrid, stepBlock>>>(d_cur, d_next, ext_n); cudaCheckError();
#else
        step<<<stepGrid, stepBlock>>>(d_cur, d_next, ext_n); cudaCheckError();
#endif

#ifdef DUMPALL
        cudaSafeCall( cudaMemcpy(cur, d_next, ext_size, cudaMemcpyDeviceToHost) );
        write_pbm(cur, ext_n, s);
#endif
        cell_t *d_tmp = d_cur;
        d_cur = d_next;
        d_next = d_tmp;
    }
    cudaDeviceSynchronize();
    const double elapsed = hpc_gettime() - tstart;
    /* Copy back result to host */
    cudaSafeCall( cudaMemcpy(cur, d_cur, ext_size, cudaMemcpyDeviceToHost) );
#endif
    write_pbm(cur, ext_n, s);
    free(cur);
#ifdef SERIAL
    free(next);
#else
    cudaFree(d_cur);
    cudaFree(d_next);
#endif
    fprintf(stderr, "Elapsed time: %f (%f Mupd/s)\n", elapsed, (n*n/1.0e6)*nsteps/elapsed);
    
    return EXIT_SUCCESS;
}
