/****************************************************************************
 *
 * mpi-first-pos.c - First occurrence of a value in a vector
 *
 * Copyright (C) 2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Prima posizione di un valore in un array
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2022-02-09

[Domanda d'esame appello 2022-02-09]

Descrivere un algoritmo parallelo basato su MPI per risolvere il
problema seguente su una architettura a memoria distribuita. È dato un
array di interi `v[0..N-1]` non vuoto di lunghezza $N$, e un intero
$k$. Vogliamo determinare la posizione (indice) della prima occorrenza
di $k$ in `v[]`; se $k$ non è presente, il risultato sarà $N$ (si noti
che gli indici di `v[]` iniziano da 0, quindi sappiamo che $N$ non può
rappresentare l'indice di un elemento valido).

Ad esempio, se `v[] = {3, 15, -1, 15, 21, 15, 7}` e `k = 15`, il
risultato deve essere 1, dato che `v[1]` è la prima occorrenza del
valore `15`. Se $k$ fosse 37 il risultato deve essere 7, dato che il
valore 37 non è presente e si deve restituire la lunghezza dell'array.

Si assuma che:

- la lunghezza $N$ dell'array sia molto maggiore del numero $P$ di
  processi MPI;

- la lunghezza $N$ dell'array sia multipla di $P$;

- inizialmente solo il processo 0 (il master) conosca i valore di $N$ e
  $k$, e il contenuto dell'array `v[]`;

- al termine dell'esecuzione il processo 0 debba ricevere il
  risultato; si ricorda che il risultato deve essere compreso tra 0 e
  $N$ (estremi inclusi), dove $N$ indica che il valore non è presente.

> Dato che all'esame non viene chiesto di scrivere codice, la domanda
> proseguiva con: "Si descriva nel modo più preciso possibile il
> comportamento di ciascun processo MPI. Non è richiesto di scrivere
> codice; è possibile usare pseudocodice e/o linguaggio naturale,
> purché la spiegazione sia il più precisa possibile. In particolare,
> si indichi quale/i funzioni MPI si userebbero nei vari passi, e
> quali buffer di memoria è necessario usare/allocare."

Per compilare:

        mpicc -std=c99 -Wall -Wpedantic mpi-first-pos.c -o mpi-first-pos

Per eseguire:

        mpirun -n 4 ./mpi-first-pos [N [k]]

## File

- [mpi-first-pos.c](mpi-first-pos.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

/**
 * Returns a random value in [a, b]
 */
int randab(int a, int b)
{
    return a + rand() % (b - a + 1);
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz, N, k, i, pos, minpos;
    int *v = NULL;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        N = atoi(argv[1]);
    } else {
        N = comm_sz * 10;
    }

    if ( argc > 2 ) {
        k = atoi(argv[2]);
    } else {
        k = randab(-1, N-1);
    }

    if ((N % comm_sz != 0) && (my_rank == 0)) {
        fprintf(stderr, "FATAL: array length (%d) must be a multiple of comm_sz (%d)\n", N, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Il master inizializza l'array */
    if ( 0 == my_rank ) {
        v = (int*)malloc(N * sizeof(*v));
        assert(v != NULL);
        printf("Before: [");
        for (i=0; i<N; i++) {
            v[i] = i;
            printf("%d ", v[i]);
        }
        printf("]\n");
    }

    /* Sebbene i valori di N e k siano passati sulla riga di comando,
       e quindi conosciuti a tutti i processi, la specifica
       dell'esercizio dice che solo il master li conosce. Di
       conseguenza, facciamo due broadcast per comunicarli anche agli
       altri processi, sebbene tecnicamente queste operazioni siano
       inutili. */
    MPI_Bcast(&N,       /* buffer       */
              1,        /* count        */
              MPI_INT,  /* datatype     */
              0,        /* root         */
              MPI_COMM_WORLD );

    MPI_Bcast(&k,       /* buffer       */
              1,        /* count        */
              MPI_INT,  /* datatype     */
              0,        /* root         */
              MPI_COMM_WORLD );

    /* Tutti i processi inizializzano il buffer locale */
    const int local_N = N / comm_sz;
    int *local_v = (int*)malloc(local_N * sizeof(*local_v));
    assert(local_v != NULL);

    /* Il master distribuisce l'array `v[]` agli altri processi */
    MPI_Scatter( v,             /* senfbuf      */
                 local_N,       /* sendcount    */
                 MPI_INT,       /* sendtype     */
                 local_v,       /* recvbuf      */
                 local_N,       /* recvcount    */
                 MPI_INT,       /* recvtype     */
                 0,             /* root         */
                 MPI_COMM_WORLD /* comm         */
                 );

    /* Ogni processo effettua la ricerca. Occorre prestare attenzione
       al fatto che, in caso di valore non trovato, i processi debbano
       comunicare il valore N al master. Solo in questo modo il master
       può determinare il valore corretto in caso di chiave non
       presente. Un ulteriore punto di attenzione è che la posizione
       da comunicare al master non è quella nell'array locale, ma
       nell'array globale. */
    i = 0;
    while (i<local_N && local_v[i] != k) {
        i++;
    }
    if (i<local_N) {
        pos = my_rank * local_N + i; /* attenzione... */
    } else {
        pos = N;
    }

    /* I processi effettuano una riduzione sul minimo di pos */
    MPI_Reduce(&pos,    /* sendbuf      */
               &minpos, /* recvbuf      */
               1,       /* count        */
               MPI_INT, /* datatype     */
               MPI_MIN, /* op           */
               0,       /* root         */
               MPI_COMM_WORLD );

    if ( 0 == my_rank ) {
        printf("Result: %d\n", minpos);
    }

    free(v);
    free(local_v);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
