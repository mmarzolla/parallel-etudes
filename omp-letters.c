/****************************************************************************
 *
 * omp-letters.c - Character counts
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
% HPC - Character counts
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-08-12

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

In this exercise you are required to modify the function
`make_hist(text, hist)` to make use of shared-memory parallelism using
OpenMP. The function takes as parameter a pointer `text` to the whole
text, represented as a zero-terminated C string, and an array
`hist[26]` of counts. The array `hist` is not initialized. At the end,
`hist[0]` contains the occurrences of the letter `a` in the text,
`hist[1]` the occurrences of the letter `b`, up to `hist[25]` that
represents the occurrences of the letter `z`.

A shared-memory parallel version can be developed as follows. The text
is partitioned into `num_threads` chunks, where `num_threads` is the
number of OpenMP threads; since the text is a character array of some
length $n$, thread $p$ can compute the extremes of its chunk as:

```C
const int from = (n * p) / num_threads;
const int to = (n * (p+1)) / num_threads;
```

Thread $p$ will then examine the block `text[from .. (to-1)]`.

You also need create a shared, two-dimensional array
`local_hist[num_threads][26]`, initially containing all zeros. Thread
$p$ operates on a different portion of the text and updates the
occurrences on the slice `local_hist[p][]` of the shared array.
Therefore, if thread $p$ sees character $x$, $x \in \{\texttt{'a'},
\ldots, \texttt{'z'}\}$, it will increment `local_hist[p][x - 'a']`.

When all threads are done, the master computes the results as the sums
of the columns of `local_hist`. In other words, the number of
occurrences of `a` is

        local_hist[0][0] + local_hist[1][0] + ... + local_hist[num_threads-1][0]

and so on.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-letters.c -o omp-letters

Run with:

        ./omp-letters < the-war-of-the-worlds.txt

## Files

* [omp-letters.c](omp-letters.c)
* [War and Peace](war-and-peace.txt) by L. Tolstoy
* [The Hound of the Baskervilles](the-hound-of-the-baskervilles.txt) by A. C. Doyle
* [The War of the Worlds](the-war-of-the-worlds.txt) by H. G. Wells

***/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

/**
 * Count occurrences of letters 'a'..'z' in `text`; uppercase
 * characters are transformed to lowercase, and all other symbols are
 * ignored. `text` must be zero-terminated. `hist` will be filled with
 * the computed counts. Returns the total number of letters found.
 */
int make_hist( const char *text, int hist[26] )
{
    int nlet = 0; /* total number of alphabetic characters processed */
    int i, j;
#ifdef SERIAL
    /* [TODO] Parallelize this function */

    /* Reset the histogram */
    for (j=0; j<26; j++) {
        hist[j] = 0;
    }
    /* Count occurrences */
    for (i=0; i<strlen(text); i++) {
        const char c = text[i];
        if (isalpha(c)) {
            nlet++;
            hist[ tolower(c) - 'a' ]++;
        }
    }
#else
    const int num_threads = omp_get_max_threads();
    int local_hist[num_threads][26]; /* one histogram per OpenMP thread */

    for (i=0; i<num_threads; i++) {
        for (j=0; j<26; j++) {
            local_hist[i][j] = 0;
        }
    }

#if __GNUC__ < 9
#pragma omp parallel default(none) reduction(+:nlet) private(i) shared(local_hist, text)
#else
#pragma omp parallel default(none) reduction(+:nlet) private(i) shared(local_hist, text, num_threads)
#endif
    {
        const int my_id = omp_get_thread_num();
        const int my_start = strlen(text) * my_id / num_threads;
        const int my_end = strlen(text) * (my_id + 1) / num_threads;
        for (i=my_start; i < my_end; i++) {
            const char c = text[i];
            if (isalpha(c)) {
                nlet++;
                local_hist[my_id][ tolower(c) - 'a' ]++;
            }
        }
    }

    /* compute the frequencies by summing the local histograms */
    for (j=0; j<26; j++) {
        int s = 0;
        for (i=0; i<num_threads; i++) {
            s += local_hist[i][j];
        }
        hist[j] = s;
    }
#endif

    return nlet;
}

/**
 * Print frequencies
 */
void print_hist( int hist[26] )
{
    int i;
    int nlet = 0;
    for (i=0; i<26; i++) {
        nlet += hist[i];
    }
    for (i=0; i<26; i++) {
        printf("%c : %8d (%6.2f%%)\n", 'a'+i, hist[i], 100.0*hist[i]/nlet);
    }
    printf("    %8d total\n", nlet);
}

int main( void )
{
    int hist[26];
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
