/****************************************************************************
 *
 * opencl-rule30.c - Rule30 Cellular Automaton with OpenCL
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
% Ultimo aggiornamento: 2021-12-03

Questo esercizio chiede di implementare mediante OpenCL l'automa
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

Il file [opencl-rule30.c](opencl-rule30.c) contiene lo scheletro di
una implementazione seriale dell'algoritmo che calcola l'evoluzione
dell'automa. Inizialmente tutte le celle sono nello stato 0, ad
eccezione di quella in posizione $N/2$ che è nello stato 1. Il
programma accetta sulla riga di comando la dimensione del dominio $N$
e il numero di passi da calcolare (se non indicati, si usano dei
valori di default). Al termine dell'esecuzione il programma produce il
file `opencl-rule30.pbm` contenente una immagine simile alla Figura 1.

![Figura 1: Evoluzione dell'automa cellulare della "regola 30" partendo da una singola cella attiva al centro](rule30.png)

Ogni riga dell'immagine rappresenta lo stato dell'automa in un dato
istante di tempo; il colore di ogni pixel indica lo stato di ciascuna
cella (1 = nero, 0 = bianco). Il tempo avanza dall'alto verso il
basso, quindi la prima riga indica lo stato al tempo 0, la seconda
riga lo stato al tempo 1 e così via.

Scopo di questo esercizio è di svilupparne una versione parallela in
cui il calcolo dei nuovi stati (cioè di ogni riga dell'immagine) venga
realizzato da work-item. In particolare, la funzione `rule30()` dovrà
essere trasformata in un kernel che viene invocato per calcolare un
passo di evoluzione dell'automa. Assumere che $N$ sia multiplo della
dimensione del workgroup.

Si consiglia di iniziare con una versione in cui si opera direttamente
sulla memoria globale senza usare memoria condivisa (`__local`);
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

![Figura 2: Copia dalla memoria globale alla memoria condivisa](opencl-rule30.png)

Aiutandoci con la Figura 2 procediamo come segue:

- `d_cur[]` è la copia nella memoria del device dello stato corrente
  dell'automa.

- Definire un kernel per riempire le ghost cell di `d_cur[]`; tale
  kernel, chiamato ad esempio `fill_ghost(...)`, deve essere eseguito
  da un unico work-item.

- Definire un secondo kernel 1D per aggiornare il dominio. Ciascun
  workgroup definisce un array `__local` chiamato ad esempio
  `buf[BLKDIM+2]`; usiamo `BLKDIM+2` elementi perché dobbiamo
  includere una ghost cell a sinistra e una a destra per calcolare lo
  stato successivo dei _BLKDIM_ elementi centrali senza dover
  ricorrere a ulteriori letture dalla memoria globale;

- Ciascun work-item determina l'indice `lindex` dell'elemento locale
  (nell'array `buf[]`), e l'indice `gindex` dell'elemento globale
  (nell'array `cur[]` in memoria globale) su cui deve
  operare. Estendiamo l'array `cur[]` con due ghost cell (una per
  estremità), come pure l'array `buf[]` in memoria condivisa. Quindi
  gli indici vanno calcolati come:
```C
      const int lindex = 1 + get_local_id(0);
      const int gindex = 1 + get_global_id(0);
```

- Ciascun work-item copia un elemento dalla memoria globale alla memoria
  locale:
```C
      buf[lindex] = cur[gindex];
```

- Il primo work-item di ciascun workgroup inizializza le ghost cell di
  `buf[]` ad esempio con:
```C
      if (0 == get_local_id(0)) {
          buf[0] = cur[gindex-1];
          buf[BLKDIM + 1] = cur[gindex + BLKDIM];
      }
```

Per generare l'evoluzione dell'automa, le nuove configurazioni vanno
trasferite dal _device_ all'_host_ al termine di ogni esecuzione del
kernel.

Per compilare:

        cc opencl-rule30.c simpleCL.c -o opencl-rule30 -lOpenCL

Per eseguire:

        ./opencl-rule30 [width [steps]]

Esempio:

        ./opencl-rule30 1024 1024

Il risultato sarà nel file `opencl-rule30.pbm`

## File

- [opencl-rule30.c](opencl-rule30.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

 ***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

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
    const char *outname = "opencl-rule30.pbm";
    FILE *out;
    int width = 1024, steps = 1024, s;
#ifdef SERIAL
    cell_t *cur, *next;
#else
    cell_t *cur;
    cl_mem d_cur, d_next;
    sclKernel step_kernel, fill_ghost_kernel;
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

    sclInitFromFile("opencl-rule30.cl");

    const int ext_width = width + 2;
    const size_t ext_size = ext_width * sizeof(*cur); /* includes ghost cells */
#ifdef SERIAL
    const int LEFT_GHOST = 0;
    const int LEFT = 1;
    const int RIGHT_GHOST = ext_width - 1;
    const int RIGHT = RIGHT_GHOST - 1;
#endif
    /* Create the output file */
    out = fopen(outname, "w");
    if ( !out ) {
        fprintf(stderr, "FATAL: cannot create file \"%s\"\n", outname);
        return EXIT_FAILURE;
    }
    fprintf(out, "P1\n");
    fprintf(out, "# produced by opencl-rule30.c\n");
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
    fill_ghost_kernel = sclCreateKernel("fill_ghost_kernel");
    step_kernel = sclCreateKernel("step_kernel");

    /* Allocate space for host copy of `cur[]` */
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);

    /* Initialize the domain */
    init_domain(cur, ext_width);

    /* Alloicate and initialize device data */
    d_cur = sclMallocCopy(ext_size, cur, CL_MEM_READ_WRITE);
    d_next = sclMalloc(ext_size, CL_MEM_READ_WRITE);

    const sclDim BLOCK = DIM2(SCL_DEFAULT_WG_SIZE2D, SCL_DEFAULT_WG_SIZE2D);
    const sclDim GRID = DIM2(sclRoundUp(width, SCL_DEFAULT_WG_SIZE2D),
                             sclRoundUp(width, SCL_DEFAULT_WG_SIZE2D));
    /* Evolve the CA */
    for (s=0; s<steps; s++) {

        /* Dump the current state to the output image */
        dump_state(out, cur, ext_width);

        /* Fill ghost cells */
        sclSetArgsEnqueueKernel(fill_ghost_kernel,
                                DIM1(1), DIM1(1),
                                ":b :d",
                                d_cur, ext_width);

        /* Compute next state */
        sclSetArgsEnqueueKernel(step_kernel,
                                GRID, BLOCK,
                                ":b :b :d",
                                d_cur, d_next, ext_width);

        sclMemcpyDeviceToHost(cur, d_next, ext_size);

        /* swap d_cur and d_next on the GPU */
        cl_mem d_tmp = d_cur;
        d_cur = d_next;
        d_next = d_tmp;
    }

    free(cur);
    sclFree(d_cur);
    sclFree(d_next);
    sclFinalize();
#endif

    fclose(out);

    return EXIT_SUCCESS;
}
