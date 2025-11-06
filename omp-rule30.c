/****************************************************************************
 *
 * omp-rule30.c - Rule30 Cellular Automaton
 *
 * Copyright (C) 2017--2025 Moreno Marzolla
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
% Rule 30 Cellular Automaton
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-11-06

In this exercise we implement the [Rule 30 Cellular
Automaton](https://en.wikipedia.org/wiki/Rule_30).

The Rule 30 CA is a 1D Cellular Autmaton (CA) that consists of an
array `x[N]` of $N$ integers that can be either 0 or 1. The state of
the CA evolves at discrete time steps: the new state of a cell depends
on the current state of itself and the left and right neighbors. We
assume cyclic boundary conditions, so that the neighbors of $x[0]$ are
$x[N-1]$ and $x[1]$, and the neighbors of $x[N-1]$ are $x[N-2]$ and
$x[0]$ (Figure 1).

![Figure 1: Rule 30 CA.](mpi-rule30-fig1.svg "Rule 30 CA.")

Given the current values $pqr$ of three adjacent cells, the new value
$q'$ of the cell in the middle is computed according to Table 1.

:Table 1: Rule 30 (■ = 1, □ = 0):

---------------------------------------- ----- ----- ----- ----- ----- ----- ----- -----
Current configuration $pqr$               ■■■   ■■□   ■□■   ■□□   □■■   □■□   □□■   □□□
New state $q'$ of the central cell         □     □     □     ■     ■     ■     ■     □
---------------------------------------- ----- ----- ----- ----- ----- ----- ----- -----

The sequence □□□■■■■□ = 00011110 on the second row is the binary
representation of decimal 30, from which the name ("Rule 30 CA").

The file [omp-rule30.c](omp-rule30.c) contains a serial program that
computes the evolution of the Rule 30 CA, assuming initial condition
where only the central cell is 1. The program accepts two optional
command line parameters: the domain size $N$ and the number of steps
_nsteps_. At the end, rank 0 produces an image `rule30.pbm` of size $N
\times \textit{nsteps}$ like the one shown in Figure 2. Each row
represents the state of the automaton at a specific time step (1 =
black, 0 = white). Time moves from top to bottom: the first line is
the initial state (time 0), the second line is the state at time 1,
and so on.

![Figure 2: Evolution of Rule 30 CA.](rule30.png "Evolution of Rule 30 CA.")

The pattern shown in Figure 2 is similar to the pattern on the [Conus
textile](https://en.wikipedia.org/wiki/Conus_textile) shell (Figure
3), a highly poisonous marine mollusk that lives in tropical seas.

![Figure 3: Conus Textile by Richard Ling, CC BY-SA 3.0,
<https://commons.wikimedia.org/w/index.php?curid=293495>.](conus-textile.jpg "The Conus Textile shell.")

The goal of this exercise is to parallelize the serial program using
OpenMP, so that the computation of each step is distributed across
multiple threads.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-rule30.c -o omp-rule30

To execute:

        ./omp-rule30 [width [steps [rule]]]

Example:

        OMP_NUM_THREADS=3 ./omp-rule30 1024 1024

The output is stored into a file `rule30.pbm`

## Files

- [omp-rule30.c](omp-rule30.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

typedef signed char cell_t;

/**
 * Given the current state of the CA, compute the next state. `ext_n`
 * is the number of cells PLUS the ghost cells. This function assumes
 * that the first and last cell of `cur` are ghost cells, and
 * therefore their values are used to compute `next` but are not
 * updated on the `next` array.
 */
void step( const cell_t *cur, cell_t *next, int ext_n, int rule )
{
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
#ifndef SERIAL
#pragma omp parallel for default(none) shared(LEFT, RIGHT, cur, next, ext_n, rule)
#endif
    for (int i = LEFT; i <= RIGHT; i++) {
        const cell_t east = cur[i-1];
        const cell_t center = cur[i];
        const cell_t west = cur[i+1];
        const int idx = (east << 2) | (center << 1) | west;
        next[i] = ((rule & (1 << idx)) != 0);
    }
}

void fill_ghost( cell_t *cur, int ext_n )
{
    cur[0] = cur[ext_n - 2];
    cur[ext_n - 1] = cur[1];
}

/**
 * Initialize the domain; all cells are 0, with the exception of a
 * single cell in the middle of the domain. `ext_n` is the width of the
 * domain PLUS the ghost cells.
 */
void init_domain( cell_t *cur, int ext_n )
{
    for (int i=0; i<ext_n; i++) {
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
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;

    for (int i=LEFT; i<=RIGHT; i++) {
        fprintf(out, "%d ", cur[i]);
    }
    fprintf(out, "\n");
}

int main( int argc, char* argv[] )
{
    const char *outname = "rule30.pbm";
    FILE *out = NULL;
    int width = 1024, nsteps = 1024, rule = 30;
    /* `cur` is the memory buffer containint `width` elements; this is
       the full state of the CA. */
    cell_t *cur = NULL, *next = NULL, *tmp;

    if ( argc > 4 ) {
        fprintf(stderr, "Usage: %s [width [nsteps [rule]]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        width = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        nsteps = atoi(argv[2]);
    }

    if (argc > 3) {
        rule = atoi(argv[3]);
    }

    /* `ext_width` is the width PLUS the halo on both sides. The halo
       is required by the serial version only; the parallel version
       would work fine with a (full) domain of length `width`, but
       would still require the halo in the local partitions. */
    const int ext_width = width + 2;

    out = fopen(outname, "w");
    if ( !out ) {
        fprintf(stderr, "FATAL: Cannot create %s\n", outname);
        return EXIT_FAILURE;
    }
    fprintf(out, "P1\n");
    fprintf(out, "# Produced by omp-rule30\n");
    fprintf(out, "%d %d\n", width, nsteps);

    cur = (cell_t*)malloc( ext_width * sizeof(*cur) ); assert(cur != NULL);
    next = (cell_t*)malloc( ext_width * sizeof(*next) ); assert(next != NULL);
    init_domain(cur, ext_width);

    for (int s=0; s<nsteps; s++) {
        /* Dump the current state to the output image */
        dump_state(out, cur, ext_width);
        fill_ghost( cur, ext_width );
        step( cur, next, ext_width, rule );
        tmp = cur;
        cur = next;
        next = tmp;
    }

    /* All done, free memory */
    free(cur);
    free(next);

    fclose(out);

    return EXIT_SUCCESS;
}
