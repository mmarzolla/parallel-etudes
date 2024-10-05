/******************************************************************************
 *
 * levenshtein.c - Levenshtein's edit distance
 *
 * Written in 2017--2022, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 ******************************************************************************/

/***
% HPC - Levenshtein's edit distance
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-08-26

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

Let $s[]$ and $t[]$ be two strings of lengths $n \leq 0, m \leq 0$
respectively. Let $L[i][j]$ be the edit distance between the prefix of
$s$ of length $i$ (denoted as $s[0..i-1]$) and the prefix of $t$ of
length $j$ (denoted as $s[0..j-1]$), $0 \leq \leq n$, $0 \leq j \leq
m$ (pay attention to the indices). In other words, $L[i][j]$ is the
minimum number of edit operations that are required to transform the
first $i$ characters of $s$ into the first $j$ characters of $t$. Each
operation is assumed to have unitary cost.

The simplest situation arises when one of the prefixes is empty,
i.e., $i=0$ or $j=0$:

- If $i=0$ the first prefix is empty, so to transform it into
  $t[0..j-1]$ we need to perform $j$ insert operations. Therefore,
  $L[0][j] = j$.

- If $j=0$ the second prefix is empty, so to transform $s[0..i-1]$
  into the empty string we need to perform $i$ removal operations.
  Therefore, $L[i][0] = i$.

If both $i$ and $j$ are nonzero, we need to look at the $i$-th
character of string $s$ ($s[i-1]$) and the $j$-th character of string
$t$ ($t[j-1]$). We have the following cases:

- If $s[i-1] = t[j-1]$, then the last character of the prefixes is the
  same. Therefore, to transform the substring $s[0..i-1]$ into
  $t[0..j-1]$ we ignore the last characters and transform $s[0..i-2]$
  into $t[0..j-2]$. The cost of the latter is $L[i-1][j-1]$ (note the
  indices). Hence, in this case: $L[i][j] = L[i-1][j-1]$.

- If $s[i-1] \neq t[j-1]$, we have three sub-choices:

  a. Delete the last character of the substring $s[i-1]$ and transform
     the rest into $t[0..j-1]$. Cost is $1 + L[i-1][j]$ (one delete
     operation, plus the cost of transforming $s[i-2]$ into $t[j-1]$).

  b. Delete the last character of the substring $t[j-1]$ and transform
     $s[0..i-1]$ into $t[0..j-2]$. Cost is $1 + L[i][j-1]$.

  c. Replace the last character of $s[0..i-1]$ with the last
     character of $t[0..j-1]$, and transform the prefix $s[0..i-2]$
     into $t[0..j-2]$. Cost is $1 + L[i-1][j-1]$.

All cases above can be summarized in a single equation:

$$
L[i][j] = \begin{cases}
j & \mbox{if $i=0, j > 0$} \\
i & \mbox{if $i > 0, j=0$} \\
1 + \min\{L[i][j-1], L[i-1][j], L[i-1][j-1] + 1_{s[i-1] = t[j-1]}\}& \mbox{if $i>0, j>0$}
\end{cases}
$$

where $1_P$ is the _indicator function_ for predicate $P$, i.e., an
expression whose value is 1 iff $P$ is true, 0 otherwise.  The result
is $L[n+1][m+1]$.

The core of the algorithm is the computation of the entries of matrix
$L[][]$ of size $(n+1) \times (m+1)$; the equation above shows that
the matrix can be filled using two nested loops, and is based on a
_three-point stencil_ since the value of each element depends of the
value above, on the left, and on the upper left corner.

Unfortunately, it is not possible to apply an `omp parallel for`
directive to either loops due to loop-carried dependences. However, we
can rewrite the loops so that the matrix is filled diagonally through
a _wavefront computation_. The computation of the values on the
diagonal can indeed be computed in parallel since they have no
inter-dependences.

Compile with:

        gcc -std=c99 -Wall -pedantic -fopenmp omp-levenshtein.c -o omp-levenshtein

Run with:

        ./levenshtein str1 str2

Example:

        ./levenshtein "prova prova" "test prova 1"

## Files

- [omp-levenshtein.c](omp-levenshtein.c)

***/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#ifndef min
int min( int a, int b )
{
    return ( a < b ? a : b );
}
#endif

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
            L[i][j] = min(min(L[i-1][j] + 1, L[i][j-1] + 1), L[i-1][j-1] + (s[i-1] != t[j-1]));
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
            L[i][j] = min(min(L[i-1][j] + 1, L[i][j-1] + 1), L[i-1][j-1] + (s[i-1] != t[j-1]));
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
