/****************************************************************************
 *
 * mpi-dynamic.c - simulate "schedule(dynamic)" using mpi
 *
 * Copyright (C) 2023 by Alice Girolomini <alice.girolomini(at)studio.unibo.it>
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
% HPC - Implementing the "schedule(dynamic)" clause by hand
% Alice Girolomini <alice.girolomini@studio.unibo.it>
% Last updated: 2023-10-30

We know that the `schedule(dynamic)` OpenMP clause dynamically assigns
iterations of a "for" loop to the first available OpenMP thread. The
purpose of this exercise is to simulate the `schedule(dynamic)` clause
using MPI in order to show an example of the master-worker pattern in 
a distributed memory model.

File [mpi-dynamic.c](mpi-dynamic.c) contains a serial program that
initializes an array `vin[]` with $n$ random integers (the value of
$n$ can be passed from the command line). The program creates a second
array `vout[]` of the same length, where `vout[i] = Fib(vin[i])` for
each $i$. `Fib(k)` is the _k_-th Fibonacci number: `Fib(0) = Fib(1) =
1`; `Fib(k) = Fib(k-1) + Fib(k-2)` for $k \geq 2$. The computation of
Fibonacci numbers is deliberately inefficient to ensure that there are
huge variations of the running time depending on the argument.

Process 0 has the role of master that splits the input array in as 
many part as specified by the constant CHUNK_SIZE. Each part
is dynamically assigned to the other processes. The workers calculate the 
local result then send it to process 0. This cycle is repeated until there's 
no chunk left.

In order to execute the parallel version the number of processes
must be > 1, otherwise there's no worker to send the tasks to.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-dynamic.c -o mpi-dynamic

To execute:

        mpirun -n P ./mpi-dynamic [n]

Example:

        mpirun -n 4 ./mpi-dynamic 10

## Files

- [mpi-dynamic.c](mpi-dynamic.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>
#include "hpc.h"

#define CHUNK_SIZE 100 /* can be set to any value >= 1 */
#define WORK_COMPLETED 1

typedef struct {
    int start, end;
    int local_vin[CHUNK_SIZE];
    int local_vout[CHUNK_SIZE];
} Work;

/**
 * Recursive computation of the n-th Fibonacci number, for n=0, 1, 2, ...
 * Do not parallelize this function. 
 */
int fib_rec (int n) {
    if (n < 2) {
        return 1;
    } else {
        return fib_rec(n - 1) + fib_rec(n - 2);
    }
}

/**
 * Iterative computation of the n-th Fibonacci number. This function
 * must be used for checking the result only. 
*/
int fib_iter (int n) {
    if (n < 2) {
        return 1;
    } else {
        int fibnm1 = 1;
        int fibnm2 = 1;
        int fibn;
        n = n - 1;
        do {
            fibn = fibnm1 + fibnm2;
            fibnm2 = fibnm1;
            fibnm1 = fibn;
            n--;
        } while (n > 0);
        return fibn;
    }
}

/**
 * Initializes the content of vector v using the values from vstart to
 * vend.  The vector is filled in such a way that there are more or
 * less the same number of contiguous occurrences of all values in
 * [vstart, vend]. 
*/
void fill (int *v, int n) {
    const int vstart = 20, vend = 35;
    const int blk = (n + vend - vstart) / (vend - vstart + 1);
    int tmp = vstart;

    for (int i = 0; i < n; i += blk) {
        for (int j = 0; j < blk && i + j < n; j++) {
            v[i + j] = tmp;
        }
        tmp++;
    }
}

int main (int argc, char* argv[]) {

    int my_rank, comm_sz, i, n = 1024;
    const int max_n = 512*1024*1024;
    int *vin, *vout;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n > max_n) {
        fprintf(stderr, "FATAL: n too large\n");
        return EXIT_FAILURE;
    }

    if (my_rank == 0) {
        /* Initializes input and output */
        vin = (int*) malloc(n * sizeof(vin[0])); 
        assert(vin != NULL);
        vout = (int*) malloc(n * sizeof(vout[0])); 
        assert(vout != NULL);

        /* Fills input array */
        for (i = 0; i < n; i++) {
            vin[i] = 25 + (i%10);
        }
    }

    const double tstart = hpc_gettime();
#ifdef SERIAL
    if (my_rank == 0) {
        for (i = 0; i < n; i++) {
            vout[i] = fib_rec(vin[i]);
            /* printf("vin[%d]=%d vout[%d]=%d\n", i, vin[i], i, vout[i]); */
        }
    }
#else
    if(comm_sz == 1){
        fprintf(stderr, "Only one process is running, switch to the serial version\n");
        return EXIT_FAILURE;
    }

    MPI_Status status;
    /* Defines the struct worktype */
    Work current_work;
    int count = 1; 
    int blklens[] = {2 + (2 * CHUNK_SIZE)};
    MPI_Aint displs[1];
    MPI_Datatype oldtypes[1] = {MPI_INT}, worktype;
    displs[0] = 0;
    MPI_Type_create_struct(count, blklens, displs, oldtypes, &worktype);
    MPI_Type_commit(&worktype); 


    if (my_rank == 0) {
        int inactive_p = 0;
        int last_work_id = 0;

        /* Process 0 calculates and distributes one task to each process */
        for (int i = 1; i < comm_sz; i++) {
            if (last_work_id < n && (n - last_work_id) >= CHUNK_SIZE) {
                current_work.start = last_work_id;
                current_work.end = last_work_id + CHUNK_SIZE;
                memcpy(current_work.local_vin,  vin + current_work.start, CHUNK_SIZE * sizeof(int));
                MPI_Send(&current_work, 1, worktype, i, 0, MPI_COMM_WORLD);
                last_work_id += CHUNK_SIZE;
            } else {
                if (last_work_id == n || (last_work_id <= n && (n - last_work_id) < CHUNK_SIZE)) {
                    MPI_Send(&current_work, 1, worktype, i, WORK_COMPLETED, MPI_COMM_WORLD);
                    inactive_p ++;
                }
            }
        }

        /* The master receives the partial result from any worker and dispatches a new task until there are no more chunks left */
        while ((n - last_work_id) >= CHUNK_SIZE) {
            MPI_Recv(&current_work, 1, worktype, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            memcpy(vout + current_work.start, current_work.local_vout, CHUNK_SIZE * sizeof(int));
           
            current_work.start = last_work_id;
            current_work.end = last_work_id + CHUNK_SIZE;
            memcpy(current_work.local_vin,  vin + current_work.start, CHUNK_SIZE * sizeof(int));
            MPI_Send(&current_work, 1, worktype, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            last_work_id += CHUNK_SIZE;
            
        }

        /* The master receives results for pending work requests and communicates all the workers to exit */
        for (int i = 1; i < comm_sz - inactive_p; i++) {
            MPI_Recv(&current_work, 1, worktype, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            memcpy(vout + current_work.start, current_work.local_vout, CHUNK_SIZE * sizeof(int));

            MPI_Send(&current_work, 1, worktype, i, WORK_COMPLETED, MPI_COMM_WORLD);
        }

        /* The master takes care of any leftover */
        if(last_work_id < n){
            for (int i = last_work_id; i < n; i++) {
                vout[i] = fib_rec(vin[i]);
            }
        }
        printf("Work completed flag sent\n");

    } else {
        int flag = 0; 

        /* The worker receives the current job and then sends the result to the master */
        while (flag != WORK_COMPLETED) {
            MPI_Recv(&current_work, 1, worktype, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            /* Checks the tag of the received message */
            if (status.MPI_TAG == WORK_COMPLETED) {
                printf("COMPLETED rank %d\n", my_rank);
                flag = WORK_COMPLETED;
            } else {
                for (int j = 0; j < CHUNK_SIZE; j++) {
                    current_work.local_vout[j] = fib_rec(current_work.local_vin[j]);
                }

                MPI_Send(&current_work, 1, worktype, 0, 0, MPI_COMM_WORLD);
            }
        }

    }
    MPI_Type_free(&worktype);
#endif

    const double elapsed = hpc_gettime() - tstart;
    if (my_rank == 0) {
        /* check result */
        for (i = 0; i < n; i++) {
            if (vout[i] != fib_iter(vin[i])) {
                fprintf(stderr, "Test FAILED: vin[%d]=%d, vout[%d]=%d (expected %d)\n", i, vin[i], i, vout[i], fib_iter(vin[i]));
                return EXIT_FAILURE;
            }
        }
        printf("Test OK\n");
        fprintf(stderr,"\nExecution time %f seconds\n", elapsed);

        free(vin);
        free(vout);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
