/****************************************************************************
 *
 * cuda-rule30.cu - Rule30 Cellular Automaton with CUDA
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
% HPC - Automa cellulare della "regola 30"
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2021-11-12

Questo esercizio chiede di implementare mediante CUDA l'automa
cellulare della [regola 30](https://en.wikipedia.org/wiki/Rule_30) che
abbiamo già incontrato in una precedente esercitazione. Per comodità
riportiamo la descrizione del funzionamento dell'automa.

L'automa è costituito da un array $x[N]$ di $N$ interi, ciascuno dei
quali può avere valore 0 oppure 1. Lo stato dell'automa evolve durante
istanti discreti nel tempo: lo stato di una cella al tempo $t$ dipende
dal suo stato e da quello dei due vicini al tempo $t-1$. Assumiamo un
dominio ciclico, per cui i vicini della cella $x[0]$ sono $x[N-1]$ e
$x[1]$ e i vicini della cella $x[N-1]$ sono $x[N-2]$ e $x[0]$.

Dati i valori correnti $pqr$ di tre celle adiacenti, il nuovo valore
$q'$ della cella centrale è determinato in base alla tabella seguente
(■ = 1, □ = 0):

:Regola 30

---------------------------------------- ----- ----- ----- ----- ----- ----- ----- -----
Configurazione corrente ($pqr$)           ■■■   ■■□   ■□■   ■□□   □■■   □■□   □□■   □□□
Nuovo stato della cella centrale ($q'$)    □     □     □     ■     ■     ■     ■     □
---------------------------------------- ----- ----- ----- ----- ----- ----- ----- -----

(si noti che la sequenza □□□■■■■□ = 00011110, che si legge sulla
seconda riga della tabella, rappresenta il numero 30 in binario, da
cui il nome "regola 30").

Il file [cuda-rule30.cu](cuda-rule30.cu) contiene lo scheletro di una
implementazione seriale dell'algoritmo che calcola l'evoluzione
dell'automa. Inizialmente tutte le celle sono nello stato 0, ad
eccezione di quella in posizione $N/2$ che è nello stato 1. Il
programma accetta sulla riga di comando la dimensione del dominio $N$
e il numero di passi da calcolare (se non indicati, si usano dei
valori di default). Al termine dell'esecuzione il programma produce il
file `cuda-rule30.pbm` contenente una immagine simile alla Figura 1.

![Figura 1: Evoluzione dell'automa cellulare della "regola 30" partendo da una singola cella attiva al centro](rule30.png)

Ogni riga dell'immagine rappresenta lo stato dell'automa in un dato
istante di tempo; il colore di ogni pixel indica lo stato di ciascuna
cella (1 = nero, 0 = bianco). Il tempo avanza dall'alto verso il
basso, quindi la prima riga indica lo stato al tempo 0, la seconda
riga lo stato al tempo 1 e così via.

Scopo di questo esercizio è di svilupparne una versione parallela in
cui il calcolo dei nuovi stati (cioè di ogni riga dell'immagine) venga
realizzato da thread CUDA. In particolare, la funzione `rule30()`
dovrà essere trasformata in un kernel che viene invocato per calcolare
un passo di evoluzione dell'automa. Assumere che $N$ sia multiplo del
numero di thread per blocco (_BLKDIM_).

Si consiglia di iniziare con una versione in cui si opera direttamente
sulla memoria globale senza usare memoria condivisa (`__shared__`);
questa prima versione si può ricavare molto velocemente partendo dal
codice seriale fornito come esempio.

Da quanto detto a lezione, l'uso della memoria condivisa può essere
utile perché ogni valore di stato nella memoria globale viene letto
più volte (tre, per la precisione). Realizzare una seconda versione
del programma che sfrutti la memoria condivisa. L'idea è la stessa
dell'esempio visto a lezione relativo ad una computazione di tipo
stencil, con la differenza che in questo caso il raggio della _stencil_
vale 1, cioè il nuovo stato di ogni cella dipende dallo stato
precedente di quella cella e dei due vicini. Si presti però attenzione
che, a differenza dell'esempio visto a lezione, qui si assume un
dominio ciclico. La parte più delicata sarà quindi la copia dei valori
dalla memoria globale alla memoria condivisa.

![Figura 2: Copia dalla memoria globale alla memoria condivisa](cuda-rule30.png)

Aiutandoci con la Figura 2 procediamo come segue:

- `d_cur[]` è la copia nella memoria del device dello stato corrente
  dell'automa.

- Definire un kernel per riempire le ghost cell di `d_cur[]`; tale
  kernel, chiamato ad esempio `fill_ghost(...)`, deve essere eseguito da
  un singolo thread e quindi andrà invocato come `fill_ghost<<<1, 1>>>(...)`

- Definire un secondo kernel 1D per aggiornare il dominio. Ciascun
  thread block definisce un array `__shared__` chiamato ad esempio
  `buf[BLKDIM+2]`; usiamo `BLKDIM+2` elementi perché dobbiamo
  includere una ghost cell a sinistra e una a destra per calcolare lo
  stato successivo dei _BLKDIM_ elementi centrali senza dover
  ricorrere a ulteriori letture dalla memoria globale;

- Ciascun thread determina l'indice lindex dell'elemento locale
  (nell'array `buf[]`), e l'indice gindex dell'elemento globale
  (nell'array `cur[]` in memoria globale) su cui deve
  operare. Estendiamo l'array `cur[]` con due ghost cell (una per
  estremità), come pure l'array `buf[]` in memoria condivisa. Quindi
  gli indici vanno calcolati come:
```C
      const int lindex = 1 + threadIdx.x;
      const int gindex = 1 + threadIdx.x + blockIdx.x * blockDim.x;
```

- Ciascun thread copia un elemento dalla memoria globale alla memoria
  condivisa:
```C
      buf[lindex] = cur[gindex];
```

- Il primo thread di ciascun blocco inizializza le ghost cell di
  `buf[]` ad esempio con:
```C
      if (0 == threadIdx.x) {
          buf[0] = cur[gindex-1];
          buf[BLKDIM + 1] = cur[gindex + BLKDIM];
      }
```

Per generare l'evoluzione dell'automa, le nuove configurazioni vanno
trasferite dal _device_ all'_host_ al termine di ogni esecuzione del
kernel.

Per compilare:

        nvcc cuda-rule30.cu -o cuda-rule30

Per eseguire:

        ./cuda-rule30 [width [steps]]

Esempio:

        ./cuda-rule30 1024 1024

Il risultato sarà nel file `cuda-rule30.pbm`

## File

- [cuda-rule30.cu](cuda-rule30.cu)
- [hpc.h](hpc.h)

 ***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef unsigned char cell_t;

#ifdef SERIAL
/**
 * Given the current state of the CA, compute the next state.  This
 * version requires that the `cur` and `next` arrays are extended with
 * ghost cells; therefore, `ext_n` is the length of `cur` and `next`
 * _including_ ghost cells.
 *
 *                             +----- ext_n-2
 *                             |   +- ext_n-1
 *   0   1                     V   V
 * +---+-------------------------+---+
 * |///|                         |///|
 * +---+-------------------------+---+
 *
 */
void step( cell_t *cur, cell_t *next, int ext_n )
{
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    int i;
    for (i=LEFT; i<=RIGHT; i++) {
        const cell_t left   = cur[i-1];
        const cell_t center = cur[i  ];
        const cell_t right  = cur[i+1];
        next[i] =
            ( left && !center && !right) ||
            (!left && !center &&  right) ||
            (!left &&  center && !right) ||
            (!left &&  center &&  right);
    }
}
#else
#define BLKDIM 1024

__device__ int d_min(int a, int b)
{
    return (a < b ? a : b);
}

/**
 * Fill ghost cells in device memory. This kernel must be launched
 * with one thread only.
 */
__global__ void fill_ghost( cell_t *cur, int ext_n )
{
    const int LEFT_GHOST = 0;
    const int LEFT = 1;
    const int RIGHT_GHOST = ext_n - 1;
    const int RIGHT = RIGHT_GHOST - 1;
    cur[RIGHT_GHOST] = cur[LEFT];
    cur[LEFT_GHOST] = cur[RIGHT];
}

/**
 * Given the current state `cur` of the CA, compute the `next`
 * state. This function requires that `cur` and `next` are extended
 * with ghost cells; therefore, `ext_n` is the lenght of `cur` and
 * `next` _including_ ghost cells.
 */
__global__ void step( cell_t *cur, cell_t *next, int ext_n )
{
    __shared__ cell_t buf[BLKDIM+2];
    const int gindex = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    const int lindex = 1 + threadIdx.x;

    if ( gindex < ext_n - 1 ) {
        buf[lindex] = cur[gindex];
        if (1 == lindex) {
            /* The thread with threadIdx.x == 0 (therefore, with
               lindex == 1) fills the two ghost cells of `buf[]` (one
               on the left, one on the right). When the width of the
               domain (ext_n - 2) is not multiple of BLKDIM, care must
               be taken. Indeed, if the width is not multiple of
               `BLKDIM`, then the rightmost ghost cell of the last
               thread block is `buf[1+len]`, where len is computed as
               follows: */
            const int len = d_min(BLKDIM, ext_n - 1 - gindex);
            buf[0] = cur[gindex - 1];
            buf[1+len] = cur[gindex + len];
        }

        __syncthreads();

        const cell_t left   = buf[lindex-1];
        const cell_t center = buf[lindex  ];
        const cell_t right  = buf[lindex+1];

        next[gindex] =
            ( left && !center && !right) ||
            (!left && !center &&  right) ||
            (!left &&  center && !right) ||
            (!left &&  center &&  right);
    }
}
#endif

/**
 * Initialize the domain; all cells are 0, with the exception of a
 * single cell in the middle of the domain. `cur` points to an array
 * of length `ext_n`; the length includes two ghost cells.
 */
void init_domain( cell_t *cur, int ext_n )
{
    int i;
    for (i=0; i<ext_n; i++) {
        cur[i] = 0;
    }
    cur[ext_n/2] = 1;
}

/**
 * Dump the current state of the CA to PBM file `out`. `cur` points to
 * an array of length `ext_n` that includes two ghost cells.
 */
void dump_state( FILE *out, const cell_t *cur, int ext_n )
{
    int i;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    for (i=LEFT; i<=RIGHT; i++) {
        fprintf(out, "%d ", cur[i]);
    }
    fprintf(out, "\n");
}

int main( int argc, char* argv[] )
{
    const char *outname = "cuda-rule30.pbm";
    FILE *out;
    int width = 1024, steps = 1024, s;
#ifdef SERIAL
    cell_t *cur, *next;
#else
    cell_t *cur;
    cell_t *d_cur, *d_next;
#endif

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [width [steps]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        width = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        steps = atoi(argv[2]);
    }

    const int ext_width = width + 2;
    const size_t ext_size = ext_width * sizeof(*cur); /* includes ghost cells */
    const int LEFT_GHOST = 0;
    const int LEFT = 1;
    const int RIGHT_GHOST = ext_width - 1;
    const int RIGHT = RIGHT_GHOST - 1;
    /* Create the output file */
    out = fopen(outname, "w");
    if ( !out ) {
        fprintf(stderr, "FATAL: cannot create file \"%s\"\n", outname);
        return EXIT_FAILURE;
    }
    fprintf(out, "P1\n");
    fprintf(out, "# produced by cuda-rule30.cu\n");
    fprintf(out, "%d %d\n", width, steps);

#ifdef SERIAL
    /* Allocate space for the `cur[]` and `next[]` arrays */
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    next = (cell_t*)malloc(ext_size); assert(next != NULL);

    /* Initialize the domain */
    init_domain(cur, ext_width);

    /* Evolve the CA */
    for (s=0; s<steps; s++) {

        /* Dump the current state */
        dump_state(out, cur, ext_width);

        /* Fill ghost cells */
        cur[RIGHT_GHOST] = cur[LEFT];
        cur[LEFT_GHOST] = cur[RIGHT];

        /* Compute next state */
        step(cur, next, ext_width);

        /* swap cur and next */
        cell_t *tmp = cur;
        cur = next;
        next = tmp;
    }

    free(cur);
    free(next);
#else
    /* Allocate space for `d_cur[]` and `d_next[]` on the device */
    cudaSafeCall( cudaMalloc((void **)&d_cur, ext_size) );
    cudaSafeCall( cudaMalloc((void **)&d_next, ext_size) );

    /* Allocate space for host copy of `cur[]` */
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);

    /* Initialize the domain */
    init_domain(cur, ext_width);

    /* Copy input to device */
    cudaSafeCall( cudaMemcpy(d_cur, cur, ext_size, cudaMemcpyHostToDevice) );

    /* Evolve the CA */
    for (s=0; s<steps; s++) {

        /* Dump the current state to the output image */
        dump_state(out, cur, ext_width);

        /* Fill ghost cells */
        fill_ghost<<<1, 1>>>(d_cur, ext_width);
        cudaCheckError();

        /* Compute next state */
        step<<<(width + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_cur, d_next, ext_width);
        cudaCheckError();
        
        cudaSafeCall( cudaMemcpy(cur, d_next, ext_size, cudaMemcpyDeviceToHost) );

        /* swap d_cur and d_next on the GPU */
        cell_t *d_tmp = d_cur;
        d_cur = d_next;
        d_next = d_tmp;
    }

    free(cur);
    cudaFree(d_cur);
    cudaFree(d_next);
#endif

    fclose(out);

    return EXIT_SUCCESS;
}
