/****************************************************************************
 *
 * mpi-letters.c - Character counts
 *
 * Copyright (C) 2018--2023 Moreno Marzolla
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
% Last updated: 2023-01-19

|[By Willi Heidelbach, CC BY 2.5, <https://commons.wikimedia.org/w/index.php?curid=1181525>](letters.jpg)

The file [mpi-letters.c](mpi-letters.c) contains a serial program that
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

To create a parallel version, you may want to create a
two-dimensional, shared array `local_hist[num_threads][26]`, where
`num_threads` is the number of OpenMP threads that are
used. Initially, the array contains all zeros; then, each OpenMP
thread $p$ operates on a different portion of the text and updates the
occurrences on the slice `local_hist[p][]` of the shared array.  When
all threads are done, the master computes the results as the sums of
the columns of `local_hist`. In other words, the number of occurrences
of `a` is

	local_hist[0][0] + local_hist[1][0] + ... + local_hist[num_threads-1][0]

Compile with:

	mpicc -std=c99 -Wall -Wpedantic mpi-letters.c -o mpi-letters

Run with:

	./mpi-letters < the-war-of-the-worlds.txt

## Files

* [mpi-letters.c](mpi-letters.c)
* [War and Peace](war-and-peace.txt) by L. Tolstoy
* [The Hound of the Baskervilles](the-hound-of-the-baskervilles.txt) by A. C. Doyle
* [The War of the Worlds](the-war-of-the-worlds.txt) by H. G. Wells

***/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>

#define ALPHA_SIZE 26

/**
 * Count occurrences of letters 'a'..'z' in `text`, which is an array
 * of characters of length `n`; uppercase characters are transformed
 * to lowercase, and all other symbols are ignored. `text` must be
 * zero-terminated. `hist` will be filled with the computed
 * counts. Returns the total number of letters found.
 */
int make_hist( const char *text, int hist[ALPHA_SIZE], int n )
{
    int nlet = 0; /* total number of alphabetic characters processed */

    /* Reset the histogram */
    for (int j=0; j<ALPHA_SIZE; j++) {
        hist[j] = 0;
    }
    /* Count occurrences */
    for (int i=0; i<n; i++) {
        const char c = text[i];
        if (isalpha(c)) {
            nlet++;
            hist[ tolower(c) - 'a' ]++;
        }
    }
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
void print_hist( int hist[ALPHA_SIZE] )
{
    int nlet = 0;
    for (int i=0; i<ALPHA_SIZE; i++) {
        nlet += hist[i];
    }
    for (int i=0; i<ALPHA_SIZE; i++) {
        const float freq = 100.0*hist[i]/nlet;
        printf("%c : %8d (%6.2f%%) ", 'a'+i, hist[i], freq);
        bar(freq, 65);
        printf("\n");
    }
    printf("    %8d total\n", nlet);
}

int main( int argc, char *argv[] )
{
    char *text = NULL, *my_text = NULL;
    int my_rank, comm_sz;
    int hist[ALPHA_SIZE], my_hist[ALPHA_SIZE];
    int text_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    const size_t size = 5*1024*1024; /* maximum text size: 5 MB */

    if ( 0 == my_rank ) {
        text = (char*)malloc(size); assert(text != NULL);
        text_len = fread(text, 1, size-1, stdin);
        text[text_len] = '\0'; /* put a termination mark at the end of the text */
    }

    const double tstart = MPI_Wtime();

    MPI_Bcast( &text_len,       /* buffer       */
               1,               /* count        */
               MPI_LONG,        /* datatype     */
               0,               /* root         */
               MPI_COMM_WORLD   /* communicator */
               );

    int sendcounts[comm_sz];
    int displs[comm_sz];
    for (int i=0; i<comm_sz; i++) {
        const int rank_start = text_len*i/comm_sz;
        const int rank_end = text_len*(i+1)/comm_sz;
        sendcounts[i] = rank_end - rank_start;
        displs[i] = rank_start;
    }

    const int my_size = sendcounts[my_rank];

    my_text = (char*)malloc(my_size); assert(my_text != NULL);

    MPI_Scatterv( text,         /* send buffer      */
                  sendcounts,   /* sendcounts       */
                  displs,       /* displacements    */
                  MPI_BYTE,     /* datatype         */
                  my_text,      /* receive buffer   */
                  my_size,      /* receive count    */
                  MPI_BYTE,     /* datatype         */
                  0,            /* root             */
                  MPI_COMM_WORLD /* communicator    */
                  );

    make_hist(my_text, my_hist, my_size);

    MPI_Reduce( my_hist,  	/* send buffer      */
                hist,        	/* receive buffer   */
                ALPHA_SIZE,     /* count            */
                MPI_INT,     	/* datatype         */
                MPI_SUM,        /* operation        */
                0,              /* destination      */
                MPI_COMM_WORLD  /* communicator     */
                );

    if ( 0 == my_rank ) {
        const double elapsed = MPI_Wtime() - tstart;
        print_hist(hist);
        fprintf(stderr, "Execution time: %f\n", elapsed);
    }

    free(text);
    free(my_text);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
