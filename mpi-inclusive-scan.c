/*****************************************************************************
 *
 * mpi-scan.c - MPI_Scan demo
 *
 * Copyright (C) 2018 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * --------------------------------------------------------------------------
 *
 * This solution uses the naive approach: node 0 (the master) collects
 * all partial results, and computes the final value without using the
 * reduction primitive.
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-scan.c -o mpi-scan
 *
 * Run with:
 * mpirun -n 4 ./mpi-scan
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void check(int *s, int n, int rank)
{
    int i;
    
    for (i = 0; i < n; i++) {
        if ( s[i] != i+1+n*rank ) {
            printf("Check failed: expected s[%d]==%d, got %d\n", i, i+1+n*rank, s[i]);
            abort();
        }
    }
    printf("Check ok!\n");
}


void fill(int* local, int n) {
	int i;
	
	for(i = 0; i < n; i++) {
		local[i] = 1;
	}
}

void clean(int* scan, int n) {
	int i;
	
	for(i = 0; i < n; i++) {
		scan[i] = 0;
	}
}

/* Compute the inclusive scan of the n-elements array v[], and store
   the result in s[]. */
void inclusive_scan(int *v, int *s, int n)
{
	s[0] = v[0];
	for (int i=1;i<n;i++) {
		s[i] = s[i-1] + v[i];
	}
}

void exclusive_scan(int *v, int *s, int n) {

	s[0] = 0;
	for (int i=1; i<n; i++) {
		s[i] = s[i-1] + v[i-1];
	}
}

void process_internal_sum(int v, int *s, int rank, int n) {
	
	for (int i=0; i<n; i++) {
		s[i] += v;
	}
}

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    int *my_local, *my_scan;
    int *scan;
    int len = 100000, my_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    int sendcounts[comm_sz];
    int displs[comm_sz];
    for (int i=0; i<comm_sz; i++) {
		const int rank_start = (len * i) / comm_sz;
		const int rank_end = len * (i + 1) / comm_sz;
        sendcounts[i] = rank_end - rank_start;
        displs[i] = rank_start;
    }
    
    const int my_size = (len+comm_sz-1)/comm_sz;
    my_len = sendcounts[my_rank];

    my_local = (int *)malloc(my_size * sizeof(*my_local));
    my_scan = (int *)malloc(my_size * sizeof(*my_scan));
    
    if ( 0 == my_rank ) {
		scan = (int *)malloc(len * sizeof(*scan));
	}
    
	fill(my_local, my_len);
	clean(my_scan, my_len);
	
	int process_sum;
	int blksum[comm_sz];
	int blksum_s[comm_sz];
    int my_blksum;

	/* Each process performs an inclusive scan of its portion of array */
	inclusive_scan(my_local,my_scan,my_len);
	process_sum = my_scan[my_len-1];

    MPI_Gather ( &process_sum,
				 1,
				 MPI_INT,
				 blksum,
				 1,
				 MPI_INT,
				 0,
				 MPI_COMM_WORLD
				 );

	/* The master performs an exclusive scan on blksum_s[] */
    if ( 0 == my_rank ) {
		exclusive_scan(blksum,blksum_s,comm_sz);
	}
    
    MPI_Scatter( blksum_s,			/* send buffer    	*/
				 1,					/* send count 	    */
				 MPI_INT,			/* datatype         */
				 &my_blksum,		/* receive buffer   */
				 1,					/* receive count    */
				 MPI_INT,			/* datatype         */
				 0,					/* root  			*/
				 MPI_COMM_WORLD		/* communicator     */
			     );
	
	/* Each process increments all values of its portion of the array */
	process_internal_sum(my_blksum,my_scan,my_rank,my_len);
	
	MPI_Gatherv( my_scan,			/* sendbuf      		*/
				 my_len,			/* count		    	*/
				 MPI_INT,			/* sent datatype 		*/
				 scan,				/* recvbuf      	 	*/
				 sendcounts,    	/* recvcount (equal to sendcounts, in this case) */
                 displs,        	/* displacements 		*/
				 MPI_INT,			/* received datatype 	*/
				 0,					/* source       	 	*/
				 MPI_COMM_WORLD		/* communicator 	 	*/
			     );
	
	if ( 0 == my_rank) {
		check(scan,len,my_rank);
		free(scan);
	}
	
    free(my_local);
    free(my_scan);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
