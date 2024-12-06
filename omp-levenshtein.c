/******************************************************************************
 *
 * omp-levenshtein.c - Levenshtein's edit distance
 *
 * Written in 2017--2022, 2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 ******************************************************************************/

/***
% HPC - Levenshtein's edit distance
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-12-05

The file [omp-levenshtein.c](omp-levenhstein.c) contains a serial
implementation of [Levenshtein's
algorithm](https://en.wikipedia.org/wiki/Levenshtein_distance) for
computing the _edit distance_ between two strings.  Levenshtein's edit
distance is a measure of similarity, and is related to the minimum
number of _edit operations_ that are required to transform one string
into another. Several types of edit operations have been considered in
the literature; in this program we consider insertion, deletion and
replacement of single characters while scanning the string from left
to right.

Levenshtein's distance can be computed using _dynamic programming_.
To solve this exercise you are not required to know the details; for
the sake of completeness (and for those who are interested), a brief
description of the algorithm is provided below.

Let $s$ and $t$ be two strings of lengths $n \geq 0, m \geq 0$
respectively. Let $L[i][j]$ be the edit distance between the prefix
$s[0 \ldots i-1]$ of length $i$ and $t[0 \ldots j-1]$ of length $j$,
$i=0, \ldots, n$, $j = 0, \ldots, m$.
In other words, $L[i][j]$ is the minimum number of edit
operations that are required to transform the first $i$ characters of
$s$ into the first $j$ characters of $t$.

The base case arises when one of the prefixes is empty, i.e., $i=0$ or
$j=0$:

- If $i=0$ then the first prefix is empty, so to transform an empty
  string into $t[0 \ldots j-1]$ we need $j$ insert operations:
  $L[0][j] = j$.

- If $j=0$ then the second prefix is empty, so to transform $s[0 \ldots
  i-1]$ into the empty string we need $i$ removal operations:
  $L[i][0] = i$.

If both $i$ and $j$ are nonzero, we  have three possibilities (see Fig. 1):

  a. Delete the last character of $s[0 \ldots i-1]$ and transform $s[0
     \ldots i-2]$ into $t[0 \ldots j-1]$. Cost: $1 + L[i-1][j]$ (one
     delete operation, plus the cost of transforming $s[i-2]$ into
     $t[j-1]$).

  b. Delete the last character of $t[0 \ldots j-1]$ and transform $s[0
     \ldots i-1]$ into $t[0 \ldots j-2]$. Cost: $1 + L[i][j-1]$.

  c. Depending on the last characters of the prefixes
     of $s$ and $t$:

     1. If the last characters are the same ($s[i-1] = t[j+1]$), then
        we may keep the last characters and transform $s[0 \ldots
        i-2]$ into $t[0 \ldots j-2]$. Cost: $L[i-1][j-1]$.

     2. If the last characters are different (i.e., $s[i-1] \neq
        t[i-1]$), we can replace $s[i-1]$ with $t[j-1]$, and transform
        $s[0 \ldots i-2]$ into $t[0 \ldots j-2]$. Cost: $1 +
        L[i-1][j-1]$.

![Figure 1: Computation of $L[i][j]$](omp-levenshtein.svg)

All cases above can be summarized in a single equation:

$$
L[i][j] = \begin{cases}
j & \mbox{if $i=0, j > 0$} \\
i & \mbox{if $i > 0, j=0$} \\
1 + \min\{L[i][j-1], L[i-1][j], L[i-1][j-1] + 1_{s[i-1] = t[j-1]}\}& \mbox{if $i>0, j>0$}
\end{cases}
$$

where $1_P$ is the _indicator function_ of predicate $P$, i.e., a
function whose value is 1 iff $P$ is true, 0 otherwise.

The core of the algorithm is the computation of the entries of matrix
$L$ of size $(n+1) \times (m+1)$; the equation above shows that the
matrix can be filled using two nested loops, and is based on a
_three-point stencil_ since the value of each element depends of the
value above, on the left, and on the upper left corner.

Unfortunately, it is not possible to apply an `omp parallel for`
directive to either loops due to loop-carried dependences. However, we
can rewrite the loops so that the matrix is filled diagonally through
a _wavefront computation_. The computation of the values on the
diagonal can indeed be computed in parallel since they have no
inter-dependences.

The wavefront computation can be implemented as follows:

```C
for (int slice=0; slice < n + m - 1; slice++) {
    const int z1 = slice < m ? 0 : slice - m + 1;
    const int z2 = slice < n ? 0 : slice - n + 1;
    for (int ii = slice - z2; ii >= z1; ii--) {
        const int jj = slice - ii;
        const int i = ii + 1;
        const int j = jj + 1;
        L[i][j] = min3(L[i-1][j] + 1,
                       L[i][j-1] + 1,
                       L[i-1][j-1] + (s[i-1] != t[j-1]));
    }
}
```

and the inner loop can be parallelized.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-levenshtein.c -o omp-levenshtein

Run with:

        ./omp-levenshtein str1 str2

Example:

        ./omp-levenshtein "this is a test" "that test is different"

## Files

- [omp-levenshtein.c](omp-levenshtein.c)

***/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

int min3( int a, int b, int c )
{
    const int minab = (a < b ? a : b);
    return (c < minab ? c : minab);
}

/* This function computes the Levenshtein edit distance between
   strings s and t. If we let n = strlen(s) and m = strlen(t), this
   function uses time O(nm) and space O(nm). */
int levenshtein(const char* s, const char* t)
{
    const int n = strlen(s), m = strlen(t);
    int (*L)[m+1] = malloc((n+1)*(m+1)*sizeof(int)); /* C99 idiom: L is of type int L[][m+1] */
    int result;

    /* degenerate cases first */
    if (n == 0) return m;
    if (m == 0) return n;

    /* Initialize the first column of L */
    for (int i = 0; i <= n; i++)
        L[i][0] = i;

    /* Initialize the first row of L */
    for (int j = 0; j <= m; j++)
        L[0][j] = j;

#ifdef SERIAL
    /* [TODO] Parallelize this */
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            L[i][j] = min3(L[i-1][j] + 1,
                           L[i][j-1] + 1,
                           L[i-1][j-1] + (s[i-1] != t[j-1]));
        }
    }
#else
    /* Fills the rest fo the matrix. */
    for (int slice=0; slice < n + m - 1; slice++) {
        const int z1 = slice < m ? 0 : slice - m + 1;
        const int z2 = slice < n ? 0 : slice - n + 1;
#pragma omp parallel for default(none) shared(slice,L,s,t,z1,z2,m)
	for (int ii = slice - z2; ii >= z1; ii--) {
            const int jj = slice - ii;
            const int i = ii + 1;
            const int j = jj + 1;
            L[i][j] = min3(L[i-1][j] + 1,
                           L[i][j-1] + 1,
                           L[i-1][j-1] + (s[i-1] != t[j-1]));
        }
    }
#endif
    result = L[n][m];
    free(L);
    return result;
}

int main( int argc, char* argv[] )
{
    if ( argc != 3 ) {
	fprintf(stderr, "Usage: %s str1 str2\n", argv[0]);
	return EXIT_FAILURE;
    }

    printf("%d\n", levenshtein(argv[1], argv[2]));
    return EXIT_SUCCESS;
}
