/****************************************************************************
 *
 * mpi-my-scatter.c - Scatter usando comunicazioni punto-punto
 *
 * Copyright (C) 2021 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Scatter usando comunicazioni punto-punto
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Ultimo aggiornamento: 2021-11-04

Implementare una funzione

        void my_Scatter(const double *sendbuf,
                        int sendcount,
                        double *recvbuf,
                        int recvcount,
                        int root)

il cui comportamento sia equivalente a

        MPI_Scatter(sendbuf, sendcount, MPI_DOUBLE,
                    recfbuf, recvcount, MPI_DOUBLE, root, MPI_COMM_WORLD)

Internamente, `my_Scatter()` deve fare uso di operazioni punto-punto
`MPI_Send()` e `MPI_Recv()`.

> **Nota**. Le implementazioni efficienti di `MPI_Scatter()` non sono
> necessariamente realizzate in questo modo (anzi, raramente lo sono).
> Lo scopo di questo esercizio è di vedere una _possibile_
> implementazione allo scopo di capire cosa fa `MPI_Scatter()`.

Per realizzare la funzione `my_Scatter()`, ogni processo determina il
proprio rango $p$ e il numero $P$ di processi MPI attivi. Il processo
`root`:

- partiziona logicamente l'array `sendbuf` in porzioni di `sendcount`
  elementi ciascuna;

- invia ciascuna porzione agli altri processi

- copia la propria porzione di array `sendbuf` nel proprio `recvbuf`

Ogni altro processo diverso da `root`

- riceve la propria porzione di array e la memorizza
  in `recvbuf`

Il file [mpi-my-scatter.c](mpi-my-scatter.c) contiene lo scheletro
della funzione `my_Scatter()`. Si chiede di completarne il corpo
usando `MPI_Send()` e `MPI_Recv()`.

Compilare con:

        mpicc -std=c99 -Wall -Wpedantic mpi-my-scatter.c -o mpi-my-scatter

Eseguire con:

        mpirun -n 4 ./mpi-my-scatter

## File

- [mpi-my-scatter.c](mpi-my-scatter.c)

***/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void my_Scatter(const double *sendbuf,
                int sendcount,
                double *recvbuf,
                int recvcount,
                int root)
{
#ifdef SERIAL
    /* [TODO] Implementare questa funzione */
#else
    int my_rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    if ( my_rank == root ) {
        int p;
        for (p=0; p<comm_sz; p++) {
            const int start = sendcount * p;
            const int dest = (p != root ? p : MPI_PROC_NULL);
            MPI_Send(sendbuf + start,   /* sendbuf      */
                     sendcount,         /* sendcount    */
                     MPI_DOUBLE,        /* datatype     */
                     dest,              /* dest         */
                     0,                 /* tag          */
                     MPI_COMM_WORLD);
        }
        /* copia locale */
        memcpy(recvbuf, sendbuf + sendcount * my_rank, sendcount * sizeof(*recvbuf));
    } else {
        MPI_Recv( recvbuf,              /* buf          */
                  sendcount,            /* count        */
                  MPI_DOUBLE,           /* datatype     */
                  root,                 /* source       */
                  MPI_ANY_TAG,          /* tag          */
                  MPI_COMM_WORLD,       /* comm         */
                  MPI_STATUS_IGNORE     /* status       */
                  );
    }
#endif
}

int main( int argc, char *argv[] )
{
    const int n = 50; /* dimensione del buffer di invio */
    double *buf = NULL, *local_buf;
    int my_rank, comm_sz;
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    const int local_n = n / comm_sz;

    if (0 == my_rank) {
        buf = (double*)malloc(n * sizeof(*buf));
        assert(buf != NULL);
        for (i=0; i<n; i++)
            buf[i] = 2*i;
    }

    local_buf = (double*)malloc(local_n * sizeof(*local_buf));
    assert(local_buf != NULL);

    my_Scatter(buf, local_n, local_buf, local_n, 0);

    printf("Process %d received: ", my_rank);
    for (i=0; i<local_n; i++) {
        printf("%f ", local_buf[i]);
    }
    printf("\n");
    free(buf);
    free(local_buf);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
