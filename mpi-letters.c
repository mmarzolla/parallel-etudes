/****************************************************************************
 *
 * mpi-letters.c - Count occurrences of letters 'a'..'z' from stdin
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

// TO DO! Changes needed to the context of the exercise
	/***
	% HPC - Frequenza di caratteri
	% Moreno Marzolla <moreno.marzolla@unibo.it>
	% Last updated: 2022-08-09

	![By Willi Heidelbach, CC BY 2.5, <https://commons.wikimedia.org/w/index.php?curid=1181525>](letters.jpg)

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

#define ALPHABET 26

/**
 * Count occurrences of letters 'a'..'z' in `text`; uppercase
 * characters are transformed to lowercase, and all other symbols are
 * ignored. `text` must be zero-terminated. `hist` will be filled with
 * the computed counts. Returns the total number of letters found.
 */
int make_hist( const char *text, int hist[ALPHABET], int n )
{
	int nlet = 0; /* total number of alphabetic characters processed */
    int i, j;

    /* Reset the histogram */
    for (j=0; j<ALPHABET; j++) {
        hist[j] = 0;
    }
    /* Count occurrences */
    for (i=0; i<n; i++) {
        const char c = text[i];
        if (isalpha(c)) {
            nlet++;
            hist[ tolower(c) - 'a' ]++;
        }
    }
	return nlet;
}


/**
 * Print frequencies
 */
void print_hist( int hist[ALPHABET] )
{
    int i;
    int nlet = 0;
    for (i=0; i<ALPHABET; i++) {
        nlet += hist[i];
    }
    for (i=0; i<ALPHABET; i++) {
        printf("%c : %8d (%6.2f%%)\n", 'a'+i, hist[i], 100.0*hist[i]/nlet);
    }
    printf("    %8d total\n", nlet);
}

int main( int argc, char *argv[] )
{
	char *text, *my_text;
	int my_rank, comm_sz;
	int hist[ALPHABET], my_hist[ALPHABET];
	long N;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
	const size_t size = 5*1024*1024; /* maximum text size: 5 MB */
	text = (char*)malloc(size); assert(text != NULL);
	
	if ( 0 == my_rank ) {
		N = fread(text, 1, size-1, stdin);
		text[N] = '\0'; /* put a termination mark at the end of the text */
	}
	
	MPI_Bcast( &N,				/* buffer     		*/
			   1,				/* count        	*/
			   MPI_LONG,		/* datatype         */
			   0,				/* root             */
		       MPI_COMM_WORLD	/* communicator     */
			   );
			   
	int sendcounts[comm_sz];
    int displs[comm_sz];
    for (int i=0; i<comm_sz; i++) {
        const int rank_start = N*i/comm_sz;
        const int rank_end = N*(i+1)/comm_sz;
        sendcounts[i] = rank_end - rank_start;
        displs[i] = rank_start;
    }
    
    const int my_size = (N + comm_sz - 1) / comm_sz;
    const int my_N = sendcounts[my_rank];
    
	my_text = (char*)malloc(my_size); assert(my_text != NULL);
	
	const double tstart = MPI_Wtime();
	
	MPI_Scatterv( text,				/* send buffer    	*/	
				  sendcounts,    	/* sendcounts 		*/
                  displs,        	/* displacements 	*/
				  MPI_BYTE,			/* datatype         */
				  my_text,			/* receive buffer   */
				  my_N,				/* receive count    */
				  MPI_BYTE,			/* datatype         */
				  0,				/* root             */
				  MPI_COMM_WORLD	/* communicator     */
			      );
    
	make_hist(my_text, my_hist, my_N);
    
    MPI_Reduce( my_hist,  		/* send buffer   	*/
                hist,        	/* receive buffer   */
                ALPHABET,       /* count            */
                MPI_INT,     	/* datatype         */
                MPI_SUM,        /* operation        */
                0,              /* destination      */
                MPI_COMM_WORLD  /* communicator     */
				);
				
	const double elapsed = MPI_Wtime() - tstart;
    
    if ( 0 == my_rank ) {
		print_hist(hist);
		
		fprintf(stderr, "Elapsed time: %f\n", elapsed);
	}
    
    free(text); 
    
    MPI_Finalize();

    return EXIT_SUCCESS;
}
