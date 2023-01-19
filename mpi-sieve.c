/****************************************************************************
 *
 * mpi-sieve.c -- Sieve of Eratosthenes
 *
 * Copyright (C) 2018--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% Last updated: 2022-08-10

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

The file [omp-sieve.c](omp-sieve.c) contains a serial program that,
given an integer $n \geq 2$, computes the number $\pi(n)$ of primes in
the set $\{2, \ldots n\}$ using the sieve of
Eratosthenes[^1]. Although the serial program could be made more
efficient both in time and space, here it is best to sacrifice
efficiency for readability. The set of unmarked numbers in $\{2,
\ldots, n\]$ is represented by the `isprime[]` array of length $n+1$;
during execution, `isprime[k]` is 0 if and only if $k$ has been
marked, i.e., has been determined to be composite ($2 \leq k \leq n$);
`isprime[0]` e `isprime[1]` are not used.

[^1]: $\pi(n)$ is also called [prime-counting
      function](https://en.wikipedia.org/wiki/Prime-counting_function)

The program contains a function `int mark(char *isprime, int from, int
to, int p)` that marks all multiples of $p$ belonging to the set
$\{\texttt{from}, \ldots \texttt{to}-1\}$. The function returns the
number of values that have been marked for the first time.

The goal is to write a parallel version of the sieve of Erathostenes;
to this aim, you might want to use the following hints.

The main program contains the following instructions:

```C
count = n - 1;
for (i=2; i*i <= n; i++) {
	if (isprime[i]) {
		count -= mark(isprime, i*i, n+1, i);
	}
}
```

To compute $\pi(n)$ we start by initializing `count` as the number of
elements in the set $\{2, \ldots n\}$; every time we mark a value for
the first time, we decrement `count` so that, at the end, we have that
$\pi(n) = \texttt{count}$.

Note that it is not possible to parallelize the _for_ loop above,
because the content of `isprime[]` is possibly modified by function
`mark()`, and this represents a _loop-carried dependency_. However, it
is possible to parallelize the body of function `mark()`. The idea is
to partition the set $[i \times i, n]$ among $P$ OpenMP threads so
that every thread will mark all multiples of $i$ that belong to its
partition.

I suggest that you start using the `omp parallel` construct (not `omp
parallel for`) and compute the bounds of each partition by hand.  It
is not trivial to do so correctly, but this is quite instructive since
during the lectures we only considered the simple case of partitioning
a range $0, \ldots, n-1$, while here the range does not start at zero.

Once you have a working parallel version, you can take the easier
route to use the `omp parallel for` directive and let the compiler
partition the iteration range for you.

Compile with:

        mpicc -std=c99 -Wall -Wpedantic -fopenmp mpi-sieve.c -o mpi-sieve

Execute with:

        ./mpi-sieve [n]

Example:

        mpirun  -n 4 ./mpi-sieve 1000

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
long mark( int *isprime, int from, int n, int p, int low, int high)
{
	long nmarked = 0l;
	int gx;
	
	for(gx = from; gx<=high; gx+=p) {
		int lx = gx - low;
		if(lx >= 0 && isprime[lx]) {
			isprime[lx] = 0;
			nmarked++;
		}
	}
	
	return nmarked;
}

void clean(int* p, int n) {
	int i;
	
	for(i = 0; i < n; i++) {
		p[i] = 1;
	}
}

int main( int argc, char *argv[] )
{
	int *isprime = NULL, *my_primes, result = 0;
	int nprimes;
	double tstart = 0, elapsed = 0;
	long n = 1000000l, i;
	long my_n;
	int my_rank, comm_sz;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }
    
    n += 1;
    
    if ( 0 == my_rank ) {
		isprime = (int*)malloc(n*sizeof(*isprime)); assert(isprime != NULL);
		/* Initially, all numbers are considered primes */
		clean(isprime, n);
	}
	
	int sendcounts[comm_sz];
    int displs[comm_sz];
    for (int i=0; i<comm_sz; i++) {
		const int rank_start = (n * i) / comm_sz;
		const int rank_end = n * (i + 1) / comm_sz;
        sendcounts[i] = rank_end - rank_start;
        displs[i] = rank_start;
    }
    
    my_n = sendcounts[my_rank];
    my_primes = (int*)malloc(my_n*sizeof(*my_primes)); assert(my_primes != NULL);
    
    const int my_low = my_rank*n/comm_sz;
    const int my_high = (my_rank+1)*n/comm_sz-1;
    
	nprimes = 0;
	
	tstart = MPI_Wtime();
	
	MPI_Scatterv( isprime, 			/* sendbuf      	 	 */
				  sendcounts,    	/* sendcounts 			 */
                  displs,        	/* displacements 		 */
				  MPI_INT,			/* sent datatype 	 	 */
				  my_primes,		/* recvbuf      	 	 */
				  my_n,				/* recvcount    	 	 */
				  MPI_INT,			/* received datatype 	 */
				  0,				/* source       	 	 */
				  MPI_COMM_WORLD 	/* communicator 	 	 */
				  );
	
	for(i=2; i*i < n; i++) {
		nprimes += mark(my_primes, i*i, my_n, i, my_low, my_high);
	}
	
	MPI_Reduce( &nprimes,  		/* send buffer           */
                &result,        /* receive buffer        */
                1,              /* count                 */
                MPI_INT,     	/* datatype              */
                MPI_SUM,        /* operation             */
                0,              /* destination           */
                MPI_COMM_WORLD  /* communicator          */
                );
    
	elapsed = MPI_Wtime() - tstart;
	
	
	if ( 0 == my_rank) {
		printf("There are %ld primes in {2, ..., %ld}\n", n-2-result, n-1);
		printf("Elapsed time: %f\n", elapsed);
	}

	free(isprime);
    free(my_primes);
    
    

    MPI_Finalize();

    return EXIT_SUCCESS;
}
