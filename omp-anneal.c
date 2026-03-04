/****************************************************************************
 *
 * omp-anneal.cu - ANNEAL cellular automaton
 *
 * Copyright (C) 2017--2026 Moreno Marzolla
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
% HPC - ANNEAL cellular automaton
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-03-04

The ANNEAL Callular Automaton (also known as _twisted majority rule_)
is a simple two-dimensional, binary CA defined on a grid of size $W
\times H$. Cyclic boundary conditions are assumed, so that each cell
has eight neighbors. Two cells are adjacent if they share a side or a
corner.

The automaton evolves at discrete time steps $0, 1, \ldots$. The new
state $x'$ at time $t+1$ of a cell $x$ depends on its current state at
time $t$ and on the current state of its neighbors. Specifically, let
$B_x$ be the number of cells in state 1 within the neighborhood of
size $3 \times 3$ centered on $x$ (including $x$ itself). Then, if
$B_x = 4$ or $B_x \geq 6$ the new state $x'$ is 1, otherwise it is 0:

$$
x' = \begin{cases}
1 & \mbox{if $B_x=4$ or $B_x \geq 6$} \\
0 & \mbox{otherwise}
\end{cases}
$$

See Figure 1 for some examples.

![Figure 1: Computation of the new state of the central cell of a block of size $3 \times 3$.](anneal.svg)

To simulate synchronous, concurrent updates of all cells, two domains
must be used. The state of a cell is read from the "current" domain,
and new values are written to the "next" domain. "Current" and "next"
are exchanged at the end of each step.

The initial states are chosen at random with uniform
probability. Figure 2 shows the evolution of a grid of size $256
\times 256$ after 10, 100 and 1024 steps. We observe the emergence of
"blobs" that grow over time, with the exception of small "specks".

![Figure 2: Evolution of the _ANNEAL_ CA.](anneal-demo.png)

I made a short YouTube video to show the evolution of the automaton
over time:

<iframe style="display:block; margin:0 auto" width="560" height="315" src="https://www.youtube-nocookie.com/embed/TSHWSjICCxs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The program [omp-anneal.c](omp-anneal.c) computes the evolution of
the _ANNEAL_ CA after $K$ iterations. The final state is written to a
file. The goal of this exercise is to parallelize the program
using OpenMP.

To compile:

        gcc -fopenmp -std=c99 -Wall -Wpedantic omp-anneal.c -o omp-anneal

To generate an image after every step:

        gcc -fopenmp -std=c99 -Wall -Wpedantic -DDUMPALL omp-anneal.c -o omp-anneal

You can make an AVI / MPEG-4 animation using:

        ffmpeg -y -i "omp-anneal-%06d.pbm" -vcodec mpeg4 omp-anneal.avi

To execute:

        ./omp-anneal [steps [W [H]]]

Example:

        ./omp-anneal 64

## References

- Tommaso Toffoli, Norman Margolus, _Cellular Automata Machines: a new
  environment for modeling_, MIT Press, 1987, ISBN 9780262526319.
  [PDF](https://people.csail.mit.edu/nhm/cam-book.pdf) from [Norman
  Margolus home page](https://people.csail.mit.edu/nhm/).

## Files

- [omp-anneal.c](omp-anneal.c)
- [hpc.h](hpc.h)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

typedef unsigned char cell_t;

/* The following function simplifies indexing of the 2D
   domain. Instead of writing grid[i*ext_width + j] you write
   IDX(grid, ext_width, i, j) to get a pointer to grid[i][j]. This
   function assumes that the size of the CA grid is
   (ext_width*ext_height), where the first and last rows/columns are
   ghost cells. */
cell_t* IDX(cell_t *grid, int ext_width, int i, int j)
{
    return (grid + i*ext_width + j);
}

/*
  `grid` points to a (ext_width * ext_height) block of bytes; this
  function copies the top and bottom ext_width elements to the
  opposite halo (see figure below).

   LEFT_GHOST=0     RIGHT=ext_width-2
   | LEFT=1         | RIGHT_GHOST=ext_width-1
   | |              | |
   v v              v v
  +-+----------------+-+
  |Y|YYYYYYYYYYYYYYYY|Y| <- TOP_GHOST=0
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- TOP=1
  | |                | |
  | |                | |
  | |                | |
  | |                | |
  |Y|YYYYYYYYYYYYYYYY|Y| <- BOTTOM=ext_height - 2
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- BOTTOM_GHOST=ext_height - 1
  +-+----------------+-+

 */
void copy_top_bottom(cell_t *grid, int ext_width, int ext_height)
{
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;

#ifndef SERIAL
    /* The iterations of this loop are independent, hence the loop can
       be parallelized. However, `ext_width` is relatively small, so
       the overhead introduced by OpenMP might exceed the benefits. */
#pragma omp parallel for
#endif
    for (int j=0; j<ext_width; j++) {
        *IDX(grid, ext_width, BOTTOM_GHOST, j) = *IDX(grid, ext_width, TOP, j); /* top to bottom halo */
        *IDX(grid, ext_width, TOP_GHOST, j) = *IDX(grid, ext_width, BOTTOM, j); /* bottom to top halo */
    }
}

/*
  `grid` points to a ext_width*ext_height block of bytes; this
  function copies the left and right ext_height elements to the
  opposite halo (see figure below).

   LEFT_GHOST=0     RIGHT=ext_width-2
   | LEFT=1         | RIGHT_GHOST=ext_width-1
   | |              | |
   v v              v v
  +-+----------------+-+
  |X|Y              X|Y| <- TOP_GHOST=0
  +-+----------------+-+
  |X|Y              X|Y| <- TOP=1
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y| <- BOTTOM=ext_height - 2
  +-+----------------+-+
  |X|Y              X|Y| <- BOTTOM_GHOST=ext_height - 1
  +-+----------------+-+

 */
void copy_left_right(cell_t *grid, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

#ifndef SERIAL
    /* The iterations of this loop are independent, hence the loop can
       be parallelized. However, `ext_width` is relatively small, so
       the overhead introduced by OpenMP might exceed the benefits. */
#pragma omp parallel for
#endif
    for (int i=0; i<ext_height; i++) {
        *IDX(grid, ext_width, i, RIGHT_GHOST) = *IDX(grid, ext_width, i, LEFT); /* left column to right halo */
        *IDX(grid, ext_width, i, LEFT_GHOST) = *IDX(grid, ext_width, i, RIGHT); /* right column to left halo */
    }
}

/* Compute the `next` grid given the current configuration `cur`.
   Both grids have (ext_width * ext_height) elements. */
void step(cell_t *cur, cell_t *next, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
#ifndef SERIAL
#pragma omp parallel for default(none) shared(TOP,BOTTOM,LEFT,RIGHT,cur,next,ext_width)
#endif
    for (int i=TOP; i <= BOTTOM; i++) {
        for (int j=LEFT; j <= RIGHT; j++) {
            int nblack = 0;
            for (int di=-1; di<=1; di++) {
                for (int dj=-1; dj<=1; dj++) {
                    nblack += *IDX(cur, ext_width, i+di, j+dj);
                }
            }
            *IDX(next, ext_width, i, j) = (nblack >= 6 || nblack == 4);
        }
    }
}

/* Initialize the current grid `cur` with alive cells with density
   `p`. Do NOT parallelize this function. */
void init( cell_t *cur, int ext_width, int ext_height, float p )
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;

    srand(1234); /* initialize PRND */
    for (int i=TOP; i <= BOTTOM; i++) {
        for (int j=LEFT; j <= RIGHT; j++) {
            *IDX(cur, ext_width, i, j) = (((float)rand())/RAND_MAX < p);
        }
    }
}

/* Write `cur` to a PBM (Portable Bitmap) file whose name is derived
   from the step number `stepno`. */
void write_pbm( cell_t *cur, int ext_width, int ext_height, int stepno )
{
    char fname[128];
    FILE *f;
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;

    snprintf(fname, sizeof(fname), "omp-anneal-%06d.pbm", stepno);

    if ((f = fopen(fname, "w")) == NULL) {
        fprintf(stderr, "Cannot open %s for writing\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by omp-anneal.c\n");
    fprintf(f, "%d %d\n", ext_width-2, ext_height-2);
    for (int i=LEFT; i<=RIGHT; i++) {
        for (int j=TOP; j<=BOTTOM; j++) {
            fprintf(f, "%d ", *IDX(cur, ext_width, i, j));
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main( int argc, char* argv[] )
{
    cell_t *cur, *next;
    int nsteps = 64, width = 512, height = 512, s;
    const int MAXN = 2048;

    if ( argc > 4 ) {
        fprintf(stderr, "Usage: %s [nsteps [W [H]]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        nsteps = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        width = height = atoi(argv[2]);
    }

    if ( argc > 3 ) {
        height = atoi(argv[3]);
    }

    if ( width > MAXN || height > MAXN ) { /* maximum image size is MAXN */
        fprintf(stderr, "FATAL: the maximum allowed grid size is %d\n", MAXN);
        return EXIT_FAILURE;
    }

    const int ext_width = width + 2;
    const int ext_height = height + 2;
    const size_t ext_size = ext_width * ext_height * sizeof(cell_t);

    fprintf(stderr, "Anneal CA: steps=%d size=%d x %d\n", nsteps, width, height);

    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    next = (cell_t*)malloc(ext_size); assert(next != NULL);
    init(cur, ext_width, ext_height, 0.5);
    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        copy_top_bottom(cur, ext_width, ext_height);
        copy_left_right(cur, ext_width, ext_height);
#ifdef DUMPALL
        write_pbm(cur, ext_width, ext_height, s);
#endif
        step(cur, next, ext_width, ext_height);
        cell_t *tmp = cur;
        cur = next;
        next = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;
    write_pbm(cur, ext_width, ext_height, s);
    free(cur);
    free(next);
    fprintf(stderr, "Execution time %.3f (%f Mops/s)\n", elapsed, (width*height/1.0e6)*nsteps/elapsed);

    return EXIT_SUCCESS;
}
