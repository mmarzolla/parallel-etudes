/****************************************************************************
 *
 * mpi-rotate-right.c - Circular rotation of an array
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
% HPC - Rotazione circolare di un array
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2022-11-09

[Domanda d'esame appello 2022-01-19]

Realizzare un programma MPI che effettua una _rotazione circolare a
destra_ di un array `v[N]`.  Dato un array $v = [v_0, v_1, \ldots,
v_{N-1}]$, la sua rotazione circolare a destra è l'array $v' =
[v_{N-1}, v_0, v_1, \ldots, v_{N-2}]$.  In altre parole, ogni elemento
di $v$ viene spostato di una posizione a destra, e l'ultimo elemento a
destra diventa il primo elemento a sinistra.

Si assuma che la lunghezza $N$ sia multipla esatta del numero $P$ di
processi MPI.

> Dato che all'esame non viene chiesto di scrivere codice, la domanda
> proseguiva con: "Si descriva nel modo più preciso possibile il
> comportamento di ciascun processo MPI. Non è richiesto di scrivere
> codice; è possibile usare pseudocodice e/o linguaggio naturale,
> purché la spiegazione sia il più precisa possibile. In particolare,
> si indichi quale/i funzioni MPI si userebbero nei vari passi, e
> quali buffer di memoria è necessario usare/allocare."

Per compilare:

        mpicc -std=c99 -Wall -Wpedantic mpi-rotate-right.c -o mpi-rotate-right

Per eseguire:

        mpirun -n 4 ./mpi-rotate-right [N]

## File

- [mpi-rotate-right.c](mpi-rotate-right.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz, N, i;
    int *v = NULL;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        N = atoi(argv[1]);
    } else {
        N = comm_sz * 10;
    }

    if ((N % comm_sz != 0) && (my_rank == 0)) {
        fprintf(stderr, "FATAL: array length (%d) must be a multiple of comm_sz (%d)\n", N, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int prev = my_rank > 0 ? my_rank - 1 : comm_sz - 1;
    const int succ = my_rank < comm_sz-1 ? my_rank + 1 : 0;

    /* Il master inizializza l'array */
    if ( 0 == my_rank ) {
        v = (int*)malloc(N * sizeof(*v));
        assert(v != NULL);
        printf("Before: [");
        for (i=0; i<N; i++) {
            v[i] = i+1;
            printf("%d ", v[i]);
        }
        printf("]\n");
    }

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

    /* Ogni processo effettua una rotazione locale; prima di questo
       occorre salvare l'ultimo elemento di local_v[], dato che andrà
       inviato al processo successivo. */
    const int last = local_v[local_N - 1];
    for (i = local_N-1; i > 0; i--) {
        local_v[i] = local_v[i-1];
    }

    /* Ogni processo invia `last` al processo successivo. Questo è il
       punto critico dell'esercizio, perché se non viene realizzato
       correttamente può causare deadlock. La comunicazione può essere
       effettuata, in ordine decrescente di preferenza:

       1. Usando MPI_Sendrecv

       2. Usando send/recv asincrone (almeno una delle due)

       3. Invertendo l'ordine di una send/recv in modo da impedire
          l'eventuale deadlock.

       Nota: la soluzione 3 andrebbe evitata perché rende il codice
       difficile da leggere e fragile.
    */
    MPI_Sendrecv(&last,         /* sendbuf      */
                 1,             /* sendcount    */
                 MPI_INT,       /* sendtype     */
                 succ,          /* dest         */
                 0,             /* sendtag      */
                 &local_v[0],   /* recvbuf      */
                 1,             /* recvcount    */
                 MPI_INT,       /* recvtype     */
                 prev,          /* source       */
                 0,             /* recvtag      */
                 MPI_COMM_WORLD, /* comm        */
                 MPI_STATUS_IGNORE /* status    */
                 );

    /* Il master riassembla le porzioni locali */
    MPI_Gather(local_v,         /* sendbuf      */
               local_N,         /* sendcount    */
               MPI_INT,         /* sendtype     */
               v,               /* recvbuf      */
               local_N,         /* recvcount    */
               MPI_INT,         /* recvtype     */
               0,               /* root         */
               MPI_COMM_WORLD   /* comm         */
               );

    if ( 0 == my_rank ) {
        printf("After: [");
        for (i=0; i<N; i++) {
            printf("%d ", v[i]);
        }
        printf("]\n");
    }

    free(v);
    free(local_v);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
