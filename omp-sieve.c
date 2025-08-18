/****************************************************************************
 *
 * omp-sieve.c - Sieve of Eratosthenes
 *
 * Copyright (C) 2018--2024 Moreno Marzolla
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

/***
% Sieve of Eratosthenes
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-10-05

![Eratosthenes (276 BC--194 BC)](Eratosthenes.png "Etching of an ancient seal identified as Eartosthenes")

The _sieve of Erathostenes_ is an algorithm for identifying the prime
numbers within the set $\{2, \ldots, n\}$. An integer $p \geq 2$ is
prime if and only if its only divisors are 1 and $p$ itself (2 is
prime).

To illustrate how the sieve of Eratosthenes works, let us consider
$n=20$. We start by listing all integers $2, \ldots n$:

![](omp-sieve1.svg)

The first value in the list (2) is prime; we mark all its multiples,
and get:

![](omp-sieve2.svg)

The next unmarked value (3) is prime. We mark all its multiples
starting from $3 \times 3$, since $3 \times 2$ has already been marked
as a multiple of two. We get:

![](omp-sieve3.svg)

The next unmarked value (5) is prime. The smaller unmarked multiple of
5 is $5 \times 5$, because $5 \times 2$, $5 \times 3$ and $5 \times 4$
have already been marked as multiples of 2 and 3. However, since $5
\times 5$ is outside the upper bound of the interval, the algorithm
terminates and all unmarked numbers are prime:

![](omp-sieve4.svg)

The file [omp-sieve.c](omp-sieve.c) contains a serial program that
takes as input an integer $n \geq 2$, and computes the number $\pi(n)$
of primes in the set $\{2, \ldots n\}$ using the sieve of
Eratosthenes[^1]. Although the serial program could be made more
efficient, for the sake of this exercise we trade efficiency for
readability.

The set of unmarked numbers in $\{2, \ldots, n\}$ is represented by
the `isprime[]` array of length $n+1$; during execution, `isprime[k]`
is 0 if and only if $k$ has been marked, i.e., has been determined to
be composite; `isprime[0]` and `isprime[1]` are not used.

[^1]: $\pi(n)$ is the [prime-counting
      function](https://en.wikipedia.org/wiki/Prime-counting_function)

The goal of this exercise is to write a parallel version of the sieve
of Erathostenes; to this aim, you might want to use the following
hints.

The function `primes()` contains the loop:

```C
nprimes = n - 1;
for (i=2; ((long)i)*i <= (long)n; i++) {
	if (isprime[i]) {
                nprimes -= mark(isprime, i, i*i, n+1);
	}
}
```

To compute $\pi(n)$ we start by initializing `nprimes` as the number
of elements in the set $\{2, \ldots n\}$; every time we mark a value
for the first time, we decrement `nprimes` so that, at the end, we
have $\pi(n) = \texttt{nprimes}$.

The function `mark()` has the following signature:

        int mark( char *isprime, int k, int from, int to )

and its purpose is to mark all multiples of `k`, starting from $k
\times k$, that belong to the set $\{\texttt{from}, \ldots,
\texttt{to}-1\}$.  The function returns the number of values that have
been marked _for the first time_.

It is not possible to parallelize the loop above, because the array
`isprime[]` is modified by the function `mark()`, and this represents
a _loop-carried dependency_. However, it is possible to parallelize
the body of function `mark()` (refer to the provided source code). The
idea is to partition the set $\{\texttt{from}, \ldots \texttt{to}-1\}$
among $P$ threads so that every thread will mark all multiples of $k$
that belong to its partition.

I suggest that you start using the `omp parallel` construct (not `omp
parallel for`) and compute the bounds of each partition by hand.  It
is not trivial to do so correctly, but this is quite instructive since
during the lectures we only considered the simple case of partitioning
a range $0, \ldots, n-1$, while here the range does not start at zero.

> **Note**: depending on your implementation, you may or may not
> encounter overflow problems of the `int` data type. For example,
> assume that you want to evaluate the expression
> `(n*my_id)/num_threads`, where all variables are of type `int`. If
> _n_ is large, then `n*my_id` might overflow even if the result of
> the expression could be represented as an `int`. A simple solution
> is to cast _n_ to `long`, so that the compiler will promote to
> `long` all other variables: `(((long)n)*my_id)/num_threads`.

Once you have a working parallel version, you can take the easier
route to use the `omp parallel for` directive and let the compiler
partition the iteration range for you.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-sieve.c -o omp-sieve

To execute:

        ./omp-sieve [n]

Example:

        OMP_NUM_THREADS=2 ./omp-sieve 1000

Table 1 shows the values of the prime-counting function $\pi(n)$ for
some $n$. Use the table to check the correctness of your
implementation.

:Table 1: Some values of the prime-counting function $\pi(n)$

          $n$        $\pi(n)$
-------------  --------------
            1               0
           10               4
          100              25
         1000             168
        10000            1229
       100000            9592
      1000000           78498
     10000000          664579
    100000000         5761455
   1000000000        50847534
-------------  --------------

## Files

- [omp-sieve.c](omp-sieve.c)

***/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Mark all mutliples of `k` in {`from`, ..., `to`-1}; return how many
   numbers have been marked for the first time. `from` does not need
   to be a multiple of `k`, although in this program it always is. */
int mark( char *isprime, int k, int from, int to )
{
    int nmarked = 0;
#ifdef SERIAL
    /* [TODO] Parallelize this function */
    from = ((from + k - 1)/k)*k; /* start from the lowest multiple of p that is >= from */
    for ( int x=from; x<to; x+=k ) {
        if (isprime[x]) {
            isprime[x] = 0;
            nmarked++;
        }
    }
#else
#if 1
    /* This solution works as follows: the interval [from, to-1] is
       partitioned into `num_threads` segments of approximately equal
       length. The starting point of each segment is then adjusted to
       the next multiple of p, before starting the "for" loop. */
    const int max_threads = omp_get_max_threads();
    int nmarked_p[max_threads];
#pragma omp parallel default(none) shared(isprime, from, to, k, nmarked_p)
    {
        const int my_id = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();
        const int n = (to - from);
        int my_from = from + (((long)n)*my_id)/num_threads;
        assert(my_from >= 0);
        my_from = ((my_from + k - 1)/k)*k; /* start from the lowest multiple of k that is >= my_from */
        const int my_to = from + (((long)n)*(my_id+1))/num_threads;
        nmarked_p[my_id] = 0;
        for ( int x=my_from; x<my_to; x+=k ) {
            if (isprime[x]) {
                isprime[x] = 0;
                nmarked_p[my_id]++;
            }
        }
    }
    /* The master computes the number of marked elements by performing
       a sum-reduction of nmarked_p[] */
    for (int i=0; i<max_threads; i++) {
        nmarked += nmarked_p[i];
    }
#else
    /* The following solution is simpler but less cache-efficient. The
       idea is to assign threads to iterations as follows (assuming 4
       threads):

       Thread 0: from,     from+4*k, from+8*k,  from+12*k, ...
       Thread 1: from+k,   from+5*k, from+9*k,  from+13*k, ...
       Thread 2: from+2*k, from+6*k, from+10*k, from+14*k, ...
       Thread 3: from+3*k, from+7*k, from+11*k, from+15*k, ...

    */
    from = ((from + k - 1)/k)*k; /* Start from the first multiple of k greater than or equal to "from" */
#pragma omp parallel default(none) shared(isprime, from, to, k,reduction(+:nmarked)
    {
        const int my_id = omp_get_thread_num();
        const int num_threads = omp_get_num_threads();
        for ( int x=from + k*my_id; x<to; x += k*num_threads ) {
            if (isprime[x]) {
                isprime[x] = 0;
                nmarked++;
            }
        }
    }
#endif
#endif
    return nmarked;
}

/* Return the number of primes in the set {2, ... n} */
int primes( int n )
{
    int nprimes = n-1;
    char *isprime = (char*)malloc(n+1); assert(isprime != NULL);

    /* Initially, all numbers are considered primes */
    for (int i=0; i<=n; i++)
        isprime[i] = 1;

    /* main iteration of the sieve; the expression i*i <= n is
       promoted to `long` to avoid overflow. */
    for (int i=2; ((long)i)*i <= (long)n; i++) {
        if (isprime[i]) {
            nprimes -= mark(isprime, i, i*i, n+1);
        }
    }
    /* Uncomment to print the list of primes */
    /*
    for (int i=2; i<=n; i++) {
        if (isprime[i]) { printf("%d ", i); }
    }
    printf("\n");
    */
    free(isprime);
    return nprimes;
}

int main( int argc, char *argv[] )
{
    int n = 1000000;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc == 2 ) {
        n = atoi(argv[1]);
    }

    if (n < 0) {
        fprintf(stderr, "FATAL: n too large\n");
        return EXIT_FAILURE;
    }

    const double tstart = omp_get_wtime();
    const int nprimes = primes(n);
    const double elapsed = omp_get_wtime() - tstart;

    printf("There are %d primes in {2, ..., %d}\n", nprimes, n);

    printf("Execution time: %f\n", elapsed);

    return EXIT_SUCCESS;
}
