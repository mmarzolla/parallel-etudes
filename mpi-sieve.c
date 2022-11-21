/****************************************************************************
 *
 * mpi-sieve.c - Sieve of Eratosthenes
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
% HPC - Sieve of Eratosthenes
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-21

!! TO DO !!

## Files

- [mpi-sieve.c](mpi-sieve.c)

***/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Mark all mutliples of `p` in the set {`from`, ..., `to`-1}; return how
   many numbers have been marked for the first time. `from` does not
   need to be a multiple of `p`. */
long mark( int *isprime, int rank, int step, long from, long to, long p)
{
	long nmarked = 0l;
	
	for( long x = from; x<=to; x+=p) {
		int my_index = x - step * rank;
		if(my_index >= 0 && my_index < step && isprime[my_index]) {
			isprime[my_index] = 0;
			nmarked++;
		}
	}	
	
	return nmarked;
}

int main( int argc, char *argv[] )
{
	int *isprime = NULL, *my_primes, result = 0;
	double tstart = 0, elapsed = 0;
	long n = 1000000l, i;
	int nprimes;
	long my_to, my_n;
	int my_rank, comm_sz;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }
    
    if ( 0 == my_rank && n % comm_sz ) {
		fprintf(stderr, "FATAL: the vector length (%ld) must be multiple of %d\n", n, comm_sz);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
    
    if ( 0 == my_rank ) {
		isprime = (int*)malloc(n*sizeof(*isprime)); assert(isprime != NULL);
		/* Initially, all numbers are considered primes */
		for (i=0; i<=n; i++) {
			isprime[i] = 1;
		}
	}
	
	my_n = n / comm_sz;
	my_primes = (int*)malloc(my_n*sizeof(*my_primes)); assert(my_primes != NULL);
	
	nprimes = 0;
	
	if ( 0 == my_rank ) {
		tstart = MPI_Wtime();
	}
	
	MPI_Scatter( isprime, 		/* sendbuf      	 	 */
				 my_n,			/* count; how many elements to send to _each_ destination */
				 MPI_INT,		/* sent datatype 	 	 */
				 my_primes,		/* recvbuf      	 	 */
				 my_n,			/* recvcount    	 	 */
				 MPI_INT,		/* received datatype 	 */
				 0,				/* source       	 	 */
				 MPI_COMM_WORLD /* communicator 	 	 */
				 );

	my_to = my_n * my_rank + my_n - 1;
	
	for(i=2; i*i < n; i++) {
		nprimes += mark(my_primes, my_rank, my_n, i*i, my_to, i);
	}
    
    MPI_Gather( my_primes,		/* sendbuf      	 	 */
				my_n,			/* sendcount    	 	 */
				MPI_INT,		/* sent datatype 	 	 */
				isprime,		/* recvbuf      	 	 */
				my_n,			/* recvcount; how many elements to received from _each_ node */
				MPI_INT,		/* received datatype 	 */
				0,				/* root (where to send)	 */
				MPI_COMM_WORLD	/* communicator 	 	 */
				);
	
	MPI_Reduce( &nprimes,  		/* send buffer           */
                &result,        /* receive buffer        */
                1,              /* count                 */
                MPI_INT,     	/* datatype              */
                MPI_SUM,        /* operation             */
                0,              /* destination           */
                MPI_COMM_WORLD  /* communicator          */
                );
    
    
    if ( 0 == my_rank) {
		elapsed = MPI_Wtime() - tstart;
		
		printf("There are %ld primes in {2, ..., %ld}\n", n-2-result, n);
		printf("Elapsed time: %f\n", elapsed);
	}


	free(isprime);
    free(my_primes);

    MPI_Finalize();

    return EXIT_SUCCESS;
}