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

The _sieve of Erathostenes_ is an algorithm for identifying the prime
numbers falling within a given range which usually is the set $\{2,
\ldots, n\}$ . A natural number $p \geq 2$ is prime if and only if the
only divisors are 1 and $p$ itself (2 is prime).

To illustrate how the sieve of Eratosthenes works, let us consider the
case $n=20$. We start by listing all integers $2, \ldots n$:

![](omp-sieve1.png)

The first value in the list (2) is prime; we mark all its multiples
and get:

![](omp-sieve2.png)

The next unmarked value (3) is again prime. We mark all its multiples
starting from $3 \times 3$ (indeed, $3 \times 2$ has been
marked at the previous step because it is a multiple of 2). We get:

![](omp-sieve3.png)

The next unmarked value (5) is prime. The smaller unmarked multiple of
5 is $5 \times 5$, because $5 \times 2$, $5 \times 3$ and $5 \times 4$
have all been marked since they are multiples of 2 and 3. However,
since $5 \times 5 > 20$ is outside the upper bound of the interval,
the algorithm terminates and all unmarked numbers are prime:

![](omp-sieve4.png)

The file [mpi-sieve.c](mpi-sieve.c) contains a serial program that,
given an integer $n \geq 2$, computes the number $\pi(n)$ of primes in
the set $\{2, \ldots n\}$ using the sieve of
Eratosthenes[^1]. Although the serial program could be made more
efficient both in time and space, here it is best to sacrifice
efficiency for readability. The set of unmarked numbers in $\{2,
\ldots, n\}$ is represented by the `isprime[]` array of length $n+1$;
during execution, `isprime[k]` is 0 if and only if $k$ has been
marked, i.e., has been determined to be composite ($2 \leq k \leq n$);
`isprime[0]` and `isprime[1]` are not used.

[^1]: $\pi(n)$ is also called [prime-counting
      function](https://en.wikipedia.org/wiki/Prime-counting_function)

The program contains a function `int mark_serial(char *isprime, int k,
int from, int to)` that marks all multiples of $k$ belonging to the set
$\{\texttt{from}, \ldots \texttt{to}-1\}$. The function returns the
number of values that have been marked for the first time.

The goal is to write a parallel version of the sieve of Erathostenes;
to this aim, you might want to use the following hints.

!! Hold hints !! 
    The main program contains the following instructions:

    ```C
    count = n - 1;
    for (i=2; i*i <= n; i++) {
        if (isprime[i]) {
            count -= mark(isprime, i, i*i, n+1);
        }
    }
    ```

    To compute $\pi(n)$ we start by initializing `count` as the number of
    elements in the set $\{2, \ldots n\}$; every time we mark a value for
    the first time, we decrement `count` so that, at the end, we have that
    $\pi(n) = \texttt{count}$.

    It is not possible to parallelize the _for_ loop above, because the
    content of `isprime[]` is possibly modified by function `mark()`, and
    this represents a _loop-carried dependency_. However, it is possible
    to parallelize the body of function `mark()`. The idea is to partition
    the set $\{\texttt{from}, \ldots \texttt{to}-1\}$ among $P$ OpenMP
    threads so that every thread will mark all multiples of $k$ that
    belong to its partition.

    I suggest that you start using the `omp parallel` construct (not `omp
    parallel for`) and compute the bounds of each partition by hand.  It
    is not trivial to do so correctly, but this is quite instructive since
    during the lectures we only considered the simple case of partitioning
    a range $0, \ldots, n-1$, while here the range does not start at zero.

    Once you have a working parallel version, you can take the easier
    route to use the `omp parallel for` directive and let the compiler
    partition the iteration range for you.

!! Possible hints !!
    In order to parallelize the serial program using MPI, it's required to
    subdivide the problem into smaller ones. Once understood the correct
    problem size for each process, the main process (proc 0) will have to
    send to each process its respective problem section.

    For the correct execution, each process, should know their respective
    starting and ending position compared to the whole problem. 
    The `mark_serial` function should be re-arranged accordingly.

    Once completed the execution of the new `mark` function, each process
    will have a partial number of primes numbers of the whole problem, for
    the program to work correctly it will be needed to accumulate all of the
    processes's solutions together.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-sieve.c -o omp-sieve

Execute with:

        ./omp-sieve [n]

Example:

        OMP_NUM_THREADS=2 ./omp-sieve 1000

As a reference, Table 1 shows the values of $\pi(n)$ for some
$n$. Use the table to check the correctness of your implementation

:Table 1: some values of the prime-counting function $\pi(n)$

          $n$                             $\pi(n)$
-------------  -----------------------------------
            1                                    0
           10                                    4
          100                                   25
         1000                                  168
        10000                                 1229
       100000                                 9592
      1000000                                78498
     10000000                               664579
    100000000                              5761455
   1000000000                             50847534
  10000000000  **Do not try on the server**: uses >10GB of RAM!!
-------------  -----------------------------------

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

/* Serial version*/
long mark_serial( char *isprime, int k, long from, long to )
{
    long nmarked = 0l;

    from = ((from + k - 1)/k)*k; /* start from the lowest multiple of p that is >= from */
    for ( long x=from; x<to; x+=k ) {
        if (isprime[x]) {
            isprime[x] = 0;
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

#ifdef SERIAL

    if ( 0 == my_rank ) {
        for (i=2; i*i<=n; i++) {
            if (isprime[i]) {
                result += mark_serial(isprime, i*i, n+1, i);
            }
        }
    }

#else

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

#endif
    
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