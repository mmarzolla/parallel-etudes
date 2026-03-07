/****************************************************************************
 *
 * omp-letters.c - Character counts
 *
 * Copyright (C) 2018--2026 Moreno Marzolla
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
% Character counts
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-03-07

![By Willi Heidelbach, CC BY 2.5, <https://commons.wikimedia.org/w/index.php?curid=1181525>.](letters.jpg)

The file [omp-letters.c](omp-letters.c) contains a serial program that
computes the number of occurrences of the letters 'a'...'z' in an
ASCII file read from standard input. No distinction is made between
upper and lowercase characters; non-letter characters are ignored. We
provide some text documents to experiment with, courtesy of the
[Project Gutenberg](https://www.gutenberg.org/). You should observe
that the character frequencies are about the same across the
documents, despite the fact that they have been written by different
authors. Indeed, the relative frequencies of characters are
language-dependent but author-independent. You are encouraged to
experiment with books in other languages, also available on [Project
Gutenberg Web site](https://www.gutenberg.org/).

The goal of this exercise is to modify the function `make_hist(text,
hist)` to use of OpenMP parallelism. The function takes as parameter a
pointer `text` to a zero-terminated string representing the document,
and an uninitialized array `hist[26]`. At the end, `hist[0]` must
contain the occurrences of the letter `a` in `text`, `hist[1]` the
occurrences of the letter `b`, up to `hist[25]` that represents the
occurrences of the letter `z`.

A possible approach is to partition the text among the OpenMP threads,
so that each thread computes the frequencies of a block of text; the
final result is the vector sum of all local histograms. There are
several ways to do that; to get a better understanding of how things
work, we start with a completely manual solution. A better approach
will described later on.

## Manual solution

This section describes a completely manual solution that only uses
`omp parallel` (not `omp parallel for`), and does not use array
reductions. This solution is rather cumbersome and not recommended in
practice, but is useful to understand how things work "under the
hood".

Since the text is a character array of length `TEXT_LEN`, thread $p$
computes the extremes of its chunk as:

```C
const int from = (TEXT_LEN * p) / num_threads;
const int to = (TEXT_LEN * (p+1)) / num_threads;
```

where `num_threads` is the size of OpenMP team. Thread $p$ will
compute the frequencies of the characters in `text[from .. (to-1)]`.

Each OpenMP thread computes the histogram of its local portion of the
document, which should be added to all other histograms to compute the
result. This is better done using array reductions, but for the sake
of understanding we proceed by hand.

Create a shared, two-dimensional array `local_hist[num_threads][26]`
initialized to zero. Thread $p$ operates on `local_hist[p][]` so that
no race conditions are possible. If thread $p$ sees character $x \in
\{\texttt{'a'}, \ldots, \texttt{'z'}\}$, it will increase the value
`local_hist[p][x - 'a']`.  When all threads are done, the master
computes the result as the column-wise sum of `local_hist`. In other
words, the number of occurrences of the character (`'a'` + _c_) is

$$
\texttt{hist}[c] = \sum_{p=0}^{\texttt{num_threads}-1} \texttt{local_hist}[p][c]
$$

Don't forget that there is a reduction on `nlet` that reports the
number of letters; this might be done using the `reduction()` clause
of the `omp for` directive.

## Using array reduction

A better solution is to rely on `omp parallel for` for split the
iterations of the `for` loop across threads, and array reductions
(available since OpenMP 4.5) to combine the local histograms.
The syntax of array reduction is as follows:

```C
#pragma omp parallel for default(none) shared(text, TEXT_LEN) \
            reduction(+:nlet) reduction(+:hist[:ALPHA_LEN])
for (int i=0; i<TEXT_LEN; i++) {
    \/\* ... \*\/
    hist[idx]++;
    nlet++;
    \/\* ... \*\/
}
```

The content of `hist[]` must be set to zero before the parallel
region. Array reduction performs `ALPHA_LEN` scalar reductions at the
end of the parallel region. Specifically, faor each $i$, $0 \leq i <
\texttt{ALPHA_LEN}$, `hist[i]` will be the sum of the local values of
`hist[i]` computed by all threads, plus the value of `hist[i]` right
before the parallel region.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-letters.c -o omp-letters

Run with:

        ./omp-letters < the-war-of-the-worlds.txt

## Files

* [omp-letters.c](omp-letters.c)
* Some sample texts (see [Project Gutenberg](https://www.gutenberg.org/) for more).
  - [War and Peace](war-and-peace.txt) by L. Tolstoy
  - [The Hound of the Baskervilles](the-hound-of-the-baskervilles.txt) by A. C. Doyle
  - [The War of the Worlds](the-war-of-the-worlds.txt) by H. G. Wells

***/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#define ALPHA_LEN 26

/**
 * Count occurrences of letters 'a'..'z' in `text`; uppercase
 * characters are transformed into lowercase, and all other symbols
 * are ignored. `text` must be zero-terminated. `hist` will be filled
 * with the computed counts. Returns the total number of letters
 * found.
 */
int make_hist( const char *text, int hist[ALPHA_LEN] )
{
    int nlet = 0; /* total number of alphabetic characters processed */
    const size_t TEXT_LEN = strlen(text);
#ifdef SERIAL
    /* [TODO] Parallelize this function */

    /* Reset the histogram */
    for (int j=0; j<ALPHA_LEN; j++) {
        hist[j] = 0;
    }

    /* Count occurrences */
    for (int i=0; i<TEXT_LEN; i++) {
        const char c = text[i];
        if (isalpha(c)) {
            const int idx = tolower(c) - 'a';
            nlet++;
            hist[idx]++;
        }
    }
#else
#if 0
    /* This version does not use OpenMP array reduction. */
    const int num_threads = omp_get_max_threads();
    int local_hist[num_threads][ALPHA_LEN]; /* one histogram per OpenMP thread */

#pragma omp parallel default(none) reduction(+:nlet) shared(local_hist, text, TEXT_LEN, num_threads)
    {
        const int my_id = omp_get_thread_num();
        const int my_start = (TEXT_LEN * my_id) / num_threads;
        const int my_end = (TEXT_LEN * (my_id + 1)) / num_threads;

        for (int j=0; j<ALPHA_LEN; j++) {
            local_hist[my_id][j] = 0;
        }

        for (int i=my_start; i < my_end; i++) {
            const char c = text[i];
            if (isalpha(c)) {
                const int idx = tolower(c) - 'a';
                nlet++;
                local_hist[my_id][idx]++;
            }
        }
    }

    /* Performs column-wise reduction of `local_hist[][]`. */
    for (int j=0; j<ALPHA_LEN; j++) {
        int s = 0;
        for (int i=0; i<num_threads; i++) {
            s += local_hist[i][j];
        }
        hist[j] = s;
    }
#else
    /* This version uses OpenMP built-in array reduction, that is
       available since OpenMP 4.5. */

    /* Reset the global histogram. */
    for (int j=0; j<ALPHA_LEN; j++) {
        hist[j] = 0;
    }

    /* Count occurrences. */
#pragma omp parallel for default(none) shared(text, TEXT_LEN) reduction(+:nlet) reduction(+:hist[:ALPHA_LEN])
    for (int i=0; i<TEXT_LEN; i++) {
        const char c = text[i];
        if (isalpha(c)) {
            const int idx = tolower(c) - 'a';
            nlet++;
            hist[idx]++;
        }
    }
#endif
#endif

    return nlet;
}

/**
 * If `freq == 100`, draw `len` caracters; otherwise, draw a fraction
 * of `len` characters proportional to `freq`.
 */
void bar( float freq, int len )
{
    for (int i=0; i<len*freq/100; i++) {
        printf("#");
    }
}

/**
 * Print frequencies
 */
void print_hist( int hist[ALPHA_LEN] )
{
    int nlet = 0;
    for (int i=0; i<ALPHA_LEN; i++) {
        nlet += hist[i];
    }
    for (int i=0; i<ALPHA_LEN; i++) {
        const float freq = 100.0*hist[i]/nlet;
        printf("%c : %8d (%6.2f%%) ", 'a'+i, hist[i], freq);
        bar(freq, 65);
        printf("\n");
    }
    printf("    %8d total\n", nlet);
}

int main( void )
{
    int hist[ALPHA_LEN];
    const size_t size = 5*1024*1024; /* maximum text size: 5 MB */
    char *text = (char*)malloc(size); assert(text != NULL);

    const size_t len = fread(text, 1, size-1, stdin);
    text[len] = '\0'; /* put a termination mark at the end of the text */
    const double tstart = omp_get_wtime();
    make_hist(text, hist);
    const double elapsed = omp_get_wtime() - tstart;
    print_hist(hist);
    fprintf(stderr, "Execution time %.3f\n", elapsed);
    free(text);
    return EXIT_SUCCESS;
}
