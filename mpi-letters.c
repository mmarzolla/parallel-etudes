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

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

/**
 * Count occurrences of letters 'a'..'z' in `text`; uppercase
 * characters are transformed to lowercase, and all other symbols are
 * ignored. `text` must be zero-terminated. `hist` will be filled with
 * the computed counts. Returns the total number of letters found.
 */
int make_hist( const char *text, int hist[26], int from, int to )
{
	int nlet = 0; /* total number of alphabetic characters processed */
    int i, j;

    /* Reset the histogram */
    for (j=0; j<26; j++) {
        hist[j] = 0;
    }
    /* Count occurrences */
    for (i=from; i<to; i++) {
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

int main( int argc, char *argv[] )
{
	char *text;
	int my_rank, comm_sz;
	int hist[26], my_hist[26];
	long my_from, my_to, my_len;
	
	const size_t size = 5*1024*1024; /* maximum text size: 5 MB */
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	
	text = (char*)malloc(size); assert(text != NULL);
	
	if ( 0 == my_rank ) {
		const size_t len = fread(text, 1, size-1, stdin);
		text[len] = '\0'; /* put a termination mark at the end of the text */

		my_len = (len+comm_sz-1) / comm_sz;
		
		MPI_Send( text,				/* data			*/
				  size,				/* count		*/
				  MPI_BYTE,			/* datatype		*/
				  my_rank+1,		/* destination	*/
				  my_rank+1,		/* tag			*/
				  MPI_COMM_WORLD	/* communicator	*/
				  );

		MPI_Send( &my_len,			/* data			*/
				  1,				/* count		*/
				  MPI_LONG,			/* datatype		*/
				  my_rank+1,		/* destination	*/
				  my_rank+1,		/* tag			*/
				  MPI_COMM_WORLD	/* communicator	*/
				  );
	}
	
	if ( 1 == my_rank ) {
		MPI_Recv( text,				/* data			*/
				  size,				/* count		*/
				  MPI_BYTE,			/* datatype		*/
				  my_rank-1,		/* root			*/
				  my_rank,			/* tag			*/
				  MPI_COMM_WORLD,	/* communicator	*/
				  MPI_STATUS_IGNORE /* status 		*/
				  );

		MPI_Recv( &my_len,			/* data			*/
				  1,				/* count		*/
				  MPI_LONG,			/* datatype		*/
				  my_rank-1,		/* root			*/
				  my_rank,			/* tag			*/
				  MPI_COMM_WORLD,	/* communicator	*/
				  MPI_STATUS_IGNORE	/* status 		*/
				  );
	}

	MPI_Bcast( text,					/* data			*/
			   size,					/* count		*/
			   MPI_BYTE,				/* datatype		*/
			   1,						/* root			*/
			   MPI_COMM_WORLD			/* communicator	*/
			   );
	
	MPI_Bcast( &my_len,					/* data			*/
			   1,						/* count		*/
			   MPI_LONG,				/* datatype		*/
			   1,						/* root			*/
			   MPI_COMM_WORLD			/* communicator	*/
			   );
	
	
	
    my_from = my_len*my_rank;
    my_to = my_from+my_len;
    
    const double tstart = MPI_Wtime();
    make_hist(text, my_hist, my_from, my_to);
    const double elapsed = MPI_Wtime() - tstart;
    
    MPI_Reduce( my_hist,  		/* send buffer           */
                hist,        	/* receive buffer        */
                26,             /* count                 */
                MPI_INT,     	/* datatype              */
                MPI_SUM,        /* operation             */
                0,              /* destination           */
                MPI_COMM_WORLD  /* communicator          */
				);
    
    if ( 0 == my_rank ) {
		print_hist(hist);
		
		fprintf(stderr, "Elapsed time: %f\n", elapsed);
	}
    
    free(text); 
    
    MPI_Finalize();

    return EXIT_SUCCESS;
}