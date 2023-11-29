/****************************************************************************
 *
 * mpi-rule30.c - Rule30 Cellular Automaton
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Rule 30 Cellular Automaton
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-11-22

We introduced Cellular Automata (CAs) as an example of _stencil
computations_. In this exercise we implement the [Rule 30 Cellular
Automaton](https://en.wikipedia.org/wiki/Rule_30).

The Rule 30 CA is a 1D cellular aotmaton that consists of an array
`x[N]` of $N$ integers that can be either 0 or 1. The state of the CA
evolves at discrete time steps: the new state of a cell depends on its
current state, and on the current state of the two neighbors. We
assume cyclic boundary conditions, so that the neighbors of $x[0]$ are
$x[N-1]$ and $x[1]$ and the neighbors of $x[N-1]$ are $x[N-2]$ and
$x[0]$ (Figure 1).

![Figure 1: Rule 30 CA](mpi-rule30-fig1.svg)

Given the current values $pqr$ of three adjacent cells, the new value
$q'$ of the middle cell is computed according to Table 1.

:Table 1: Rule 30 (■ = 1, □ = 0):

---------------------------------------- ----- ----- ----- ----- ----- ----- ----- -----
Current configuration $pqr$               ■■■   ■■□   ■□■   ■□□   □■■   □■□   □□■   □□□
New state $q'$ of the central cell         □     □     □     ■     ■     ■     ■     □
---------------------------------------- ----- ----- ----- ----- ----- ----- ----- -----

The sequence □□□■■■■□ = 00011110 on the second row is the binary
representation of decimal 30, from which the name ("Rule 30 CA"); more
details can be found [here](mpi-rule30.pdf).

The file [mpi-rule30.c](mpi-rule30.c) contains a serial program that
computes the evolution of the Rule 30 CA, assuming an initial
condition where all cells are 0 except the central one. The program
accepts two optional command line parameters: the domain size $N$ and
the number of steps _nsteps_ to simulate. At the end, process 0
produces the image `rule30.pbm` shown in Figure 2 of size $N \times
\textit{nsteps}$.

![Figure 2: Evolution of Rule 30 CA](rule30.png)

Each row of the image represents the state of the automaton at a
specific time step (1 = black, 0 = white). Time moves from top to
bottom: the first line is the initial state (time 0), the second line
is the state at time 1, and so on.

Interestingly, the pattern shown in Figure 2 is similar to the pattern
on the [Conus textile](https://en.wikipedia.org/wiki/Conus_textile)
shell, a highly poisonous marine mollusk which can be found in
tropical seas (Figure 3).

![Figure 3: Conus Textile by Richard Ling - Own work; Location: Cod
Hole, Great Barrier Reef, Australia, CC BY-SA 3.0,
<https://commons.wikimedia.org/w/index.php?curid=293495>](conus-textile.jpg)

The goal of this exercise is to parallelize the serial program using
MPI, so that the computation of each step is distributed across the
MPI processes. The program should operate as follows (see Figure 4):

![Figure 4: Parallelization of the Rule 30 CA](mpi-rule30-fig4.svg)

1. The domain is distributed across the $P$ MPI processes using
   `MPI_Scatter()` (assume that $N$ is a multiple of $P$). Each
   partition is augmented with two ghost cells that are required to
   compute the next states.

2. Each process sends the values on the border of its partition to the
   neighbors using `MPI_Sendrecv()`. This operation must be performed
   twice, to fill the left and right ghost cells (see below).

3. Each process computes the next state for the cells in its
   partition.

4. Since we want to dump the state after each step, the master
   collects all updated partitions using `MPI_Gather()`, and stores
   the resulting row to the output file.

At the end of step 4 we go back to step 2: since each process already
has its own updated partition, there is no need to perform a new
`MPI_Scatter()` every time.

Let `comm_sz` be the number of MPI processes.  Each partition at step
1 has ($\mathit{local\_width} = N / \mathit{comm\_sz} + 2$) elements
(including the ghosts cell). We denote with `cur[]` the complete
domain stored on process 0, and with `local_cur[]` the partitions
assigned to each process.

Assuming that `cur[]` is also extended with two ghost cells, as in the
program provided (this is not required in the MPI version), the
distribution of `cur[]` can be accomplished with the instruction:

```C
MPI_Scatter( &cur[LEFT],        \/\* sendbuf    \*\/
             N/comm_sz,         \/\* Sendcount  \*\/
             MPI_INT,           \/\* sendtype   \*\/
             &local_cur[LOCAL_LEFT],\/\* recvbuf    \*\/
             N/comm_sz,         \/\* recvcount  \*\/
             MPI_INT,           \/\* recvtype   \*\/
             0,                 \/\* root       \*\/
             MPI_COMM_WORLD     \/\* comm       \*\/
             );
```

(the symbols `LEFT` and `LOCAL_LEFT` are defined in the source code to
improve readability).

![Figure 5: Using `MPI_Sendrecv()` to exchange ghost cells](mpi-rule30-fig5.svg)

Filling the ghost cells is a bit tricky and requires two calls to
`MPI_Sendrecv()` (see Figure 5). First, each process sends the value
of the _rightmost_ domain cell to the successor, and receives the
value of the left ghost cells from the predecessor. Then, each process
sends the contents of the _leftmost_ cell to the predecessr, and
receives the value of the right ghost cell from the successor. All
processes execute the exact same type of communication: therefore,
each one should call `MPI_Sendrecv()` twice in the same way (possibly
with different parameters).

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-rule30.c -o mpi-rule30

To execute:

        mpirun -n P ./mpi-rule30 [width [steps]]

Example:

        mpirun -n 4 ./mpi-rule30 1024 1024

The output is stored to a file `rule30.pbm`

## Files

- [mpi-rule30.c](mpi-rule30.c)
- [Additional information](mpi-rule30.pdf)

***/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

/* Note: the MPI datatype corresponding to "signed char" is MPI_CHAR */
typedef signed char cell_t;

/* number of ghost cells on each side; this program assumes HALO ==
   1. */
const int HALO = 1;

/* To make the code more readable, in the following we make frequent
   use of the following variables:

    LEFT_GHOST = index of first element of left halo
          LEFT = index of first element of actual domain
         RIGHT = index of last element of actual domain
   RIGHT_GHOST = index of first element of right halo

    LEFT_GHOST                    RIGHT_GHOST
    | LEFT                            RIGHT |
    | |                                   | |
    V V                                   V V
   +-+-------------------------------------+-+
   |X| | | ...                         | | |X|
   +-+-------------------------------------+-+

     ^--------------- n -------------------^
   ^---------------- ext_n ------------------^

   We use the "LOCAL_" prefix to denote local domains, i.e., the
   portions of the domains that are stored within each MPI process.
*/

/**
 * Given the current state of the CA, compute the next state. `ext_n`
 * is the number of cells PLUS the ghost cells. This function assumes
 * that the first and last cell of `cur` are ghost cells, and
 * therefore their values are used to compute `next` but are not
 * updated on the `next` array.
 */
void step( const cell_t *cur, cell_t *next, int ext_n )
{
    int i;
    const int LEFT = HALO;
    const int RIGHT = ext_n - HALO - 1;
    for (i = LEFT; i <= RIGHT; i++) {
        const cell_t east = cur[i-1];
        const cell_t center = cur[i];
        const cell_t west = cur[i+1];
        next[i] = ( (east && !center && !west) ||
                    (!east && !center && west) ||
                    (!east && center && !west) ||
                    (!east && center && west) );
    }
}

/**
 * Initialize the domain; all cells are 0, with the exception of a
 * single cell in the middle of the domain. `ext_n` is the width of the
 * domain PLUS the ghost cells.
 */
void init_domain( cell_t *cur, int ext_n )
{
    int i;
    for (i=0; i<ext_n; i++) {
        cur[i] = 0;
    }
    cur[ext_n/2] = 1;
}

/**
 * Dump the current state of the automaton to PBM file `out`. `ext_n`
 * is the true width of the domain PLUS the ghost cells.
 */
void dump_state( FILE *out, const cell_t *cur, int ext_n )
{
    int i;
    const int LEFT = HALO;
    const int RIGHT = ext_n - HALO - 1;

    for (i=LEFT; i<=RIGHT; i++) {
        fprintf(out, "%d ", cur[i]);
    }
    fprintf(out, "\n");
}

int main( int argc, char* argv[] )
{
    const char *outname = "rule30.pbm";
    FILE *out = NULL;
    int width = 1024, nsteps = 1024, s;
    /* `cur` is the memory buffer containint `width` elements; this is
       the full state of the CA. */
    cell_t *cur = NULL, *tmp;
#ifdef SERIAL
    cell_t *next = NULL; /* This is not required by the parallel version */
#endif
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( 0 == my_rank && argc > 3 ) {
        fprintf(stderr, "Usage: %s [width [nsteps]]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if ( argc > 1 ) {
        width = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        nsteps = atoi(argv[2]);
    }

    if ( (0 == my_rank) && (width % comm_sz) ) {
        printf("The image width (%d) must be a multiple of comm_sz (%d)\n", width, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* `ext_width` is the width PLUS the halo on both sides. The halo
       is required by the serial version only; the parallel version
       would work fine with a (full) domain of length `width`, but
       would still require the halo in the local partitions. */
    const int ext_width = width + 2*HALO;

    /* The master creates the output file */
    if ( 0 == my_rank ) {
        out = fopen(outname, "w");
        if ( !out ) {
            fprintf(stderr, "FATAL: Cannot create %s\n", outname);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fprintf(out, "P1\n");
        fprintf(out, "# Produced by mpi-rule30\n");
        fprintf(out, "%d %d\n", width, nsteps);

        /* Initialize the domain

           NOTE: the parallel version does not need ghost cells in the
           cur[] array, but only in the local_cur[] blocks that are
           stored within each MPI process. For simplicity we keep the
           ghost cells in cur[]; after getting a working version,
           modify your program to remove them. */
        cur = (cell_t*)malloc( ext_width * sizeof(*cur) ); assert(cur != NULL);
#ifdef SERIAL
        /* Note: the parallel version does not need the `next`
           array. */
        next = (cell_t*)malloc( ext_width * sizeof(*next) ); assert(next != NULL);
#endif
        init_domain(cur, ext_width);
    }

    /* compute the rank of the next and previous process on the
       chain. These will be used to exchange the boundary */
#ifndef SERIAL
    const int rank_next = (my_rank + 1) % comm_sz;
    const int rank_prev = (my_rank - 1 + comm_sz) % comm_sz;
#else
    /*
    const int rank_next = ...
    const int rank_prev = ...
    */
#endif

    /* compute the size of each local domain; this should be set to
       `width / comm_sz + 2*HALO`, since it must include the ghost
       cells */
#ifndef SERIAL
    const int local_width = width / comm_sz;
    const int local_ext_width = local_width + 2*HALO;
#else
    /*
      const int local_width = ...
      const int local_ext_width = ...
    */
#endif

    /* `local_cur` and `local_next` are the local domains, handled by
       each MPI process. They both have `local_ext_width` elements each */
#ifndef SERIAL
    cell_t *local_cur = (cell_t*)malloc(local_ext_width * sizeof(*local_cur)); assert(local_cur != NULL);
    cell_t *local_next = (cell_t*)malloc(local_ext_width * sizeof(*local_next)); assert(local_next != NULL);
#else
    /*
      cell_t *local_cur = ...
      cell_t *local_next = ...
    */
#endif

    const int LEFT_GHOST = 0;
    const int LEFT = LEFT_GHOST + HALO;
#ifdef SERIAL
    const int RIGHT = ext_width - 1 - HALO;
    const int RIGHT_GHOST = RIGHT + HALO;
#endif

    /* The master distributes the domain cur[] to the other MPI
       processes. Each process receives `width/comm_sz` elements of
       type MPI_INT. Note: the parallel version does not require ghost
       cells in cur[], so it would be possible to allocate exactly
       `width` elements in cur[]. */
#ifndef SERIAL
    const int LOCAL_LEFT_GHOST = 0;
    const int LOCAL_LEFT = LOCAL_LEFT_GHOST + HALO;
    const int LOCAL_RIGHT = local_ext_width - 1 - HALO;
    const int LOCAL_RIGHT_GHOST = LOCAL_RIGHT + HALO;

    MPI_Scatter( &cur[LEFT],            /* sendbuf      */
                 local_width,           /* sendcount    */
                 MPI_CHAR,              /* datatype     */
                 &local_cur[LOCAL_LEFT],/* recvbuf      */
                 local_width,           /* recvcount    */
                 MPI_CHAR,              /* datatype     */
                 0,                     /* root         */
                 MPI_COMM_WORLD
                 );
#else
    /*
      const int LOCAL_LEFT_GHOST = ...
      const int LOCAL_LEFT = ...
      const int LOCAL_RIGHT = ...
      const int LOCAL_RIGHT_GHOST = ...

      MPI_Scatter( sendbuf,
                   sendcount,
                   datatype,
                   recvbuf,
                   recvcount,
                   datatype,
                   root,
                   MPI_COMM_WORLD
      );
    */
#endif

    for (s=0; s<nsteps; s++) {

        /* This is OK; the master dumps the current state of the automaton */
        if ( 0 == my_rank ) {
            /* Dump the current state to the output image */
            dump_state(out, cur, ext_width);
        }

        /* Send right boundary to right neighbor; receive left
           boundary from left neighbor (X=halo)

                 _________          _________
                /         V        /         V
           ...---+-+     +-+--------+-+     +-+---...
                 |X|     |X|        |X|     |X|
           ...---+-+     +-+--------+-+     +-+---...
                           local_cur

        */
#ifndef SERIAL
        MPI_Sendrecv( &local_cur[LOCAL_RIGHT], /* sendbuf */
                      HALO,             /* sendcount    */
                      MPI_CHAR,         /* datatype     */
                      rank_next,        /* dest         */
                      0,                /* sendtag      */
                      &local_cur[LOCAL_LEFT_GHOST],/* recvbuf      */
                      HALO,             /* recvcount    */
                      MPI_CHAR,         /* datatype     */
                      rank_prev,        /* source       */
                      0,                /* recvtag      */
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE
                      );
#else
        /*
        MPI_Sendrecv( sendbuf,
                      sendcount,
                      datatype,
                      dest,
                      sendtag,
                      recvbuf,
                      recvcount,
                      datatype,
                      source,
                      recvtag,
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE
                      );
        */
#endif

        /* send left boundary to left neighbor; receive right boundary
           from right neighbor

                   _________          _________
                  V         \        V         \
           ...---+-+     +-+--------+-+     +-+---...
                 |X|     |X|        |X|     |X|
           ...---+-+     +-+--------+-+     +-+---...
                           local_cur

        */
#ifndef SERIAL
        MPI_Sendrecv( &local_cur[LOCAL_LEFT], /* sendbuf      */
                      HALO,             /* sendcount    */
                      MPI_CHAR,         /* datatype     */
                      rank_prev,        /* dest         */
                      0,                /* sendtag      */
                      &local_cur[LOCAL_RIGHT_GHOST], /* recvbuf */
                      HALO,             /* recvcount    */
                      MPI_CHAR,         /* datatype     */
                      rank_next,        /* source       */
                      0,                /* recvtag      */
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE
                      );
#else
        /*
        MPI_Sendrecv( sendbuf,
                      sendcount,
                      datatype,
                      dest,
                      sendtag,
                      recvbuf,
                      recvcount,
                      datatype,
                      source,
                      recvtag,
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE
                      );
        */
#endif

#ifndef SERIAL
        step( local_cur, local_next, local_ext_width );
#else
        /* [TODO] in the parallel version, all processes must execute
           the "step()" function on ther local domains */
        if (0 == my_rank) {
            cur[LEFT_GHOST] = cur[RIGHT];
            cur[RIGHT_GHOST] = cur[LEFT];
            step(cur, next, ext_width);
        }
#endif

        /* Gather the updated local domains at the root; it is
           possible to gather the result at cur[] instead than next[];
           actually, in the parallel version, next[] is not needed at
           all. */
#ifndef SERIAL
        MPI_Gather( &local_next[LOCAL_LEFT],/* sendbuf      */
                    local_width,        /* sendcount    */
                    MPI_CHAR,           /* datatype     */
                    &cur[LEFT],         /* recvbuf      */
                    local_width,        /* recvcount    */
                    MPI_CHAR,           /* datatype     */
                    0,                  /* root         */
                    MPI_COMM_WORLD
                    );
#else
        /*
        MPI_Gather( sendbuf,
                    sendcount,
                    datatype,
                    recvbuf,
                    recvcount,
                    datatype,
                    root,
                    MPI_COMM_WORLD
                    );
        */
#endif

        /* swap current and next domain */
#ifndef SERIAL
        tmp = local_cur;
        local_cur = local_next;
        local_next = tmp;
#else
        /*
          [TODO] swap local_cur and local_next
         */

        if (0 == my_rank) {
            tmp = cur;
            cur = next;
            next = tmp;
        }
#endif
    }

    /* All done, free memory */
#ifndef SERIAL
    free(local_cur);
    free(local_next);
#else
    free(next);
#endif
    free(cur);

    if ( 0 == my_rank ) {
        fclose(out);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
