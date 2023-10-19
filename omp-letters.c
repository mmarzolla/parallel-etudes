/****************************************************************************
 *
 * omp-letters.c - Character counts
 *
 * Copyright (C) 2018--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Character counts
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-10-19

![By Willi Heidelbach, CC BY 2.5, <https://commons.wikimedia.org/w/index.php?curid=1181525>](letters.jpg)

The file [omp-letters.c](omp-letters.c) contains a serial program that
computes the number of occurrences of each lowercase letter in an
ASCII file read from standard input. The program is case-insensitive,
meaning that uppercase characters are treated as if they were
lowercase; non-letter characters are ignored. We provide some
substantial ASCII documents to experiment with, that have been made
available by the [Project Gutenberg](https://www.gutenberg.org/);
despite the fact that these documents have been written by different
authors, the frequencies of characters are quite similar. Indeed, it
is well known that the relative frequencies of characters are
language-dependent and more or less author-independent. You may
experiment with other free books in other languages that are available
on [Project Gutenberg Web site](https://www.gutenberg.org/).

The goal of this exercise is to modify the function `make_hist(text,
hist)` to make use of OpenMP parallelism. The function takes as
parameter a pointer `text` to the whole text, represented as a
zero-terminated C string, and an array `hist[26]` of counts. The array
`hist` is not initialized. At the end, `hist[0]` contains the
occurrences of the letter `a` in the text, `hist[1]` the occurrences
of the letter `b`, up to `hist[25]` that represents the occurrences of
the letter `z`.

A reasonable approach is to partition the text among the OpenMP
threads, so that each thread computes the histogram for part of the
text. Then, all partial histograms needs to be combined to get the
final result.

You might want to start by doing the partitioning manually, i.e.,
without using the `omp for` directive. This is not the most efficient
solution, but is nevertheless instructive; a better approach is
discussed below.

Since the text is a character array of some length $n$, thread $p$ can
compute the extremes of its chunk as:

```C
const int from = (n * p) / num_threads;
const int to = (n * (p+1)) / num_threads;
```

where `num_threads` is the size of OpenMP thread pool. Thread $p$ will
compute the frequencies of the characters in `text[from .. (to-1)]`.

You need to create a shared, two-dimensional array
`local_hist[num_threads][26]` initialized to zero. Thread $p$ operates
on `local_hist[p][]` so that no race conditions are possible. If
thread $p$ sees character $x$, $x \in \{\texttt{'a'}, \ldots,
\texttt{'z'}\}$, it will increase the value `local_hist[p][x - 'a']`.
When all threads are done, the master computes the result as the
column-wise sum of `local_hist`. In other words, the number of
occurrences of character `a` is

$$
\sum_{p=0}^{\texttt{num_threads}-1} \texttt{local_hist}[p][0]
$$

and so on. Also, don't forget that there is a reduction on the counter
`nlet` that reports the number of letters; this might be done using
the `reduction()` clause of the `omp for` directive.

A better and simpler solution can be realized using the `omp parallel
for` directive, and employing array reductions that are available with
OpenMP 4.5 and later (and that we did not discuss during the
class). To perform the sum-reduction on each element of the array
`hist[]` we can use the following syntax:

        #pragma omp parallel for ... reduction(+:hist[:ALPHA_SIZE])

This works as the normal scalar reductions, with the differences that
the compiler actually computes `ALPHA_SIZE` sum-reductions on
`hist[0]`, ... `hist[ALPHA_SIZE - 1]`.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-letters.c -o omp-letters

Run with:

        ./omp-letters < the-war-of-the-worlds.txt

## Files

* [omp-letters.c](omp-letters.c)
* Some sample texts
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

#define ALPHA_SIZE 26

/**
 * Count occurrences of letters 'a'..'z' in `text`; uppercase
 * characters are transformed into lowercase, and all other symbols
 * are ignored. `text` must be zero-terminated. `hist` will be filled
 * with the computed counts. Returns the total number of letters
 * found.
 */
int make_hist( const char *text, int hist[ALPHA_SIZE] )
{
    int nlet = 0; /* total number of alphabetic characters processed */
    const size_t TEXT_LEN = strlen(text);
    int i, j;
#ifdef SERIAL
    /* [TODO] Parallelize this function */

    /* Reset the histogram */
    for (j=0; j<ALPHA_SIZE; j++) {
        hist[j] = 0;
    }

    /* Count occurrences */
    for (i=0; i<TEXT_LEN; i++) {
        const char c = text[i];
        if (isalpha(c)) {
            nlet++;
            hist[ tolower(c) - 'a' ]++;
        }
    }
#else
#if 0
    /* This version does not use array reductions */
    const int num_threads = omp_get_max_threads();
    int local_hist[num_threads][ALPHA_SIZE]; /* one histogram per OpenMP thread */

    /* The following loop(s) could be parallelized, but they are quite
       short and it is unlikely that this would lead to significant
       performance improvements. */
    for (i=0; i<num_threads; i++) {
        for (j=0; j<ALPHA_SIZE; j++) {
            local_hist[i][j] = 0;
        }
    }

#pragma omp parallel default(none) reduction(+:nlet) private(i) shared(local_hist, text, TEXT_LEN, num_threads)
    {
        const int my_id = omp_get_thread_num();
        const int my_start = (TEXT_LEN * my_id) / num_threads;
        const int my_end = (TEXT_LEN * (my_id + 1)) / num_threads;
        for (i=my_start; i < my_end; i++) {
            const char c = text[i];
            if (isalpha(c)) {
                nlet++;
                local_hist[my_id][ tolower(c) - 'a' ]++;
            }
        }
    }

    /* compute the frequencies by summing the local histograms */
    for (j=0; j<ALPHA_SIZE; j++) {
        int s = 0;
        for (i=0; i<num_threads; i++) {
            s += local_hist[i][j];
        }
        hist[j] = s;
    }
#else
    /* This version uses array reductions, that is available since
       OpenMP 4.5 */
    /* Reset the histogram */
    for (j=0; j<ALPHA_SIZE; j++) {
        hist[j] = 0;
    }

    /* Count occurrences */
#pragma omp parallel for default(none) shared(text, TEXT_LEN) \
    reduction(+:nlet) \
    reduction(+:hist[:ALPHA_SIZE])
    for (i=0; i<TEXT_LEN; i++) {
        const char c = text[i];
        if (isalpha(c)) {
            nlet++;
            hist[ tolower(c) - 'a' ]++;
        }
    }
#endif
#endif

    return nlet;
}

/**
 * Print frequencies
 */
void print_hist( int hist[ALPHA_SIZE] )
{
    int i;
    int nlet = 0;
    for (i=0; i<ALPHA_SIZE; i++) {
        nlet += hist[i];
    }
    for (i=0; i<ALPHA_SIZE; i++) {
        printf("%c : %8d (%6.2f%%)\n", 'a'+i, hist[i], 100.0*hist[i]/nlet);
    }
    printf("    %8d total\n", nlet);
}

int main( void )
{
    int hist[ALPHA_SIZE];
    const size_t size = 5*1024*1024; /* maximum text size: 5 MB */
    char *text = (char*)malloc(size); assert(text != NULL);

    const size_t len = fread(text, 1, size-1, stdin);
    text[len] = '\0'; /* put a termination mark at the end of the text */
    const double tstart = omp_get_wtime();
    make_hist(text, hist);
    const double elapsed = omp_get_wtime() - tstart;
    print_hist(hist);
    fprintf(stderr, "Elapsed time: %f\n", elapsed);
    free(text);
    return EXIT_SUCCESS;
}
