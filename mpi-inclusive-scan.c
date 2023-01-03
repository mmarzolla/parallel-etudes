/****************************************************************************
 *
 * mpi-inclusive-scan.c - Inclusive Scan
 *
 * Copyright (C) 2016--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Inclusive Scan
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-21
The file [mpi-inclusive-scan.c](mpi-inclusive-scan.c) contains a MPI program
with two solutions to an inclusive scan.
The first one is a naive approach: node 0 (the master) collects all partial
results, and computes the final value without using the reduction primitive.
The second one uses the MPI_Scan directive, used to perform an inclusive
prefix reduction on data distributed across the calling processes. The operation
returns, in the `recvbuf` of the process with rank `i`, the reduction of the
values in the `sendbufs` of processses with ranks 0, ..., i.
To use the naive implementation, compile with:
        mpicc -DNAIVE -std=c99 -Wall -Wpedantic mpi-inclusive-scan.c -o mpi-inclusive-scan -lm 
To use the MPI_Scan directive, compile with:
        mpicc -std=c99 -Wall -Wpedantic mpi-inclusive-scan.c -o mpi-inclusive-scan -lm
Execute with:
        mpirun -n P ./mpi-inclusive-scan
Example:
        mpirun -n 4 ./mpi-inclusive-scan
## Files
- [mpi-inclusive-scan.c](mpi-inclusive-scan.c)
***/

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

void sum(int* local, int* scan, int n) {
	int i;
	
	scan[0] = local[0];
	for(i = 1; i < n; i++) {
		scan[i] = scan[i-1] + local[i];
	}
}

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    int *my_local, *my_scan;
    int *scan;
    int end_value = 0;
    int len = 100000, my_len;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    my_len = (len+comm_sz-1)/comm_sz;

    my_local = (int *)malloc(my_len * sizeof(*my_local));
    my_scan = (int *)malloc(my_len * sizeof(*my_scan));
    
    if ( 0 == my_rank ) {
		scan = (int *)malloc((len+comm_sz) * sizeof(*scan));
	}
    
	fill(my_local, my_len);
	clean(my_scan, my_len);
	
	if ( my_rank > 0 ) {
		MPI_Recv( &end_value,		/* buffer     		*/
				  1,				/* count        	*/
				  MPI_INT,			/* datatype         */
				  my_rank-1,		/* source           */
				  my_rank,			/* tag              */
				  MPI_COMM_WORLD,	/* communicator     */
				  &status			/* status	        */
		);

		my_local[0] += end_value;
	}
	
	sum(my_local, my_scan, my_len);
    
    if ( my_rank < comm_sz - 1 ) {
		end_value = my_scan[my_len-1];
		
		MPI_Send( &end_value,		/* buffer     		*/
				  1,				/* count        	*/
				  MPI_INT,			/* datatype         */
				  my_rank+1,		/* dest             */
				  my_rank+1,		/* tag              */
				  MPI_COMM_WORLD	/* communicator     */
		);
	}
	
	MPI_Gather( my_scan,			/* sendbuf      		*/
				my_len,				/* count		    	*/
				MPI_INT,			/* sent datatype 		*/
				scan,				/* recvbuf      	 	*/
				my_len,				/* recvcount    	 	*/
				MPI_INT,			/* received datatype 	*/
				0,					/* source       	 	*/
				MPI_COMM_WORLD		/* communicator 	 	*/
			    );
	
	if ( 0 == my_rank) {
		check(scan,len,my_rank);
	}
    
    if ( 0 == my_rank ) {
		free(scan);
	}
	
    free(my_local);
    free(my_scan);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
