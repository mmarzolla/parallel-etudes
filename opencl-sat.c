/****************************************************************************
 *
 * opencl-sat.c - Brute-force SAT solver
 *
 * Copyright (C) 2018, 2023, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Brute-force SAT solver
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last modified: 2024-09-29

To compile:

        gcc -Wall -Wpedantic opencl-sat.c simpleCL.c -o opencl-sat -LOpenCL

To execute:

        ./opencl-sat < sat.cnf

## Files

- [opencl-sat.c](opencl-sat.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h).
- Some input files: <queens-05.cnf>, <uf20-01.cnf>, <uf20-077.cnf>7

***/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include "simpleCL.h"
#include "hpc.h"

#define MAXLITERALS (8*sizeof(int) - 2)
#define MAXCLAUSES 512

typedef struct {
    int x[MAXCLAUSES], nx[MAXCLAUSES];
    int nlit;
    int nclauses;
} problem_t;

#ifndef SERIAL
sclKernel eval_kernel;
#endif

int max(int a, int b)
{
    return (a>b ? a : b);
}

int abs(int x)
{
    return (x>=0 ? x : -x);
}

#ifdef SERIAL
/**
 * Evaluate problem `p` in conjunctive normal form by setting the i-th
 * variable to the value of bit (i+1) of `v` (bit 0 is the leftmost
 * bit, which is not used). Returns the value of the boolean
 * expression encoded by `p`.
 */
bool eval(const problem_t* p, const int v)
{
    bool result = true;
    for (int c=0; c < p->nclauses && result; c++) {
        const bool term = (v & p->x[c]) | (~v & p->nx[c]);
        result &= term;
    }
    return result;
}

/**
 * Returns the number of solutions to the SAT problem `p`.
 */
int sat( const problem_t *p)
{
    const int NLIT = p->nlit;
    const int MAX_VALUE = (1 << NLIT) - 1;
    int nsat = 0;

    for (int cur_value=0; cur_value<=MAX_VALUE; cur_value++) {
        nsat += eval(p, cur_value);
    }
    return nsat;
}
#else
/**
 * OpenCL implementation of a brute-force SAT solver. It uses 1D
 * workgroup of 1D work-items; each workgroup has `p->nclauses`
 * work-items and evaluates a clause. Different work-items evaluate
 * different clauses in parallel. We can not launch `MAX_VALUE`
 * work-items (one for each possible combination of assignments),
 * since that might exceed hardware limits. Therefore, multiple kernel
 * launches are required in the "for" loop below.
 */
int sat( const problem_t *p)
{
    const int NLIT = p->nlit;
    const int NCLAUSES = p->nclauses;
    const int MAX_VALUE = (1 << NLIT) - 1;
    const sclDim BLOCK = DIM1(SCL_DEFAULT_WG_SIZE);
    const int CHUNK_SIZE = SCL_DEFAULT_WG_SIZE * 2048; /* you might need to change this depending on your hardware */
    const sclDim GRID = DIM1(CHUNK_SIZE);

    int nsat = 0;
    cl_mem d_nsat, d_x, d_nx;

    d_x = sclMallocCopy(NCLAUSES * sizeof(*(p->x)), (void*)(p->x), CL_MEM_READ_ONLY);
    d_nx = sclMallocCopy(NCLAUSES * sizeof(*(p->nx)), (void*)(p->nx), CL_MEM_READ_ONLY);
    d_nsat = sclMallocCopy(sizeof(nsat), &nsat, CL_MEM_READ_WRITE);

    for (int cur_value=0; cur_value<=MAX_VALUE; cur_value += CHUNK_SIZE) {
        sclSetArgsEnqueueKernel(eval_kernel,
                                GRID, BLOCK,
                                ":b :b :d :d :d :b",
                                d_x,
                                d_nx,
                                p->nlit,
                                p->nclauses,
                                cur_value,
                                d_nsat);
    }
    sclMemcpyDeviceToHost(&nsat, d_nsat, sizeof(nsat));

    sclFree(d_nsat);
    sclFree(d_x);
    sclFree(d_nx);
    return nsat;
}
#endif

/**
 * Pretty-prints problem `p`
 */
void pretty_print( const problem_t *p )
{
    for (int c=0; c < p->nclauses; c++) {
        printf("( ");
        int x = p->x[c];
        int nx = p->nx[c];
        for (int l=0, printed=0; l < MAXLITERALS; l++) {
            if (x & 1) {
                printf("%sx_%d", printed ? " ∨ " : "", l);
                printed = 1;
            } else if (nx & 1) {
                printf("%s¬x_%d", printed ? " ∨ " : "", l);
                printed = 1;
            }
            x = x >> 1;
            nx = nx >> 1;
        }
        printf(" )");
        if (c < p->nclauses - 1) {
            printf(" ∧");
        }
        printf("\n");
    }
}

/**
 * Load a DIMACS CNF file `f` and initialize problem `p`.  The DIMACS
 * CNF format specification can be found at
 * <https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/satformat.ps>
 */
void load_dimacs( FILE *f, problem_t *p )
{
    int result;
    int c, l, val;
    int prob_c, prob_l;

    /* Clear all bitmasks */
    for (c=0; c<MAXCLAUSES; c++) {
        p->x[c] = p->nx[c] = 0;
    }
    p->nlit = -1;
    c = l = 0;
    /* From
       <https://github.com/marijnheule/march-SAT-solver/blob/master/parser.c> */
    do {
        result = fscanf(f, " p cnf %i %i \n", &prob_l, &prob_c);
        if ( result > 0 && result != EOF )
            break;
        result = fscanf(f, "%*s\n");
    } while( result != 2 && result != EOF );

    if ( prob_l > MAXLITERALS-1 ) {
        fprintf(stderr, "FATAL: too many literals (%d); please set MAXLITERALS to at least %d\n", prob_l, prob_l+1);
        exit(EXIT_FAILURE);
    }
    if ( prob_c > MAXCLAUSES-1 ) {
        fprintf(stderr, "FATAL: too many clauses (%d); please set MAXCLAUSES to at least %d\n", prob_c, prob_c+1);
        exit(EXIT_FAILURE);
    }
    while (fscanf(f, "%d", &val) == 1) {
        if (val == 0) {
            /* Check the previous clause for consistency */
            assert( (p->x[c] & p->nx[c]) == 0 );
            /* New clause */
            l = 0;
            c++;
        } else {
            /* New literal */
            if (val > 0) {
                p->x[c] |= (1 << (val-1));
            } else {
                p->nx[c] |= (1 << -(val+1));
            }
            p->nlit = max(p->nlit, abs(val));
            l++;
        }
    }
    p->nclauses = c;
    fprintf(stderr, "DIMACS CNF files: %d clauses, %d literals\n", c, p->nlit);
}

int main( int argc, char *argv[] )
{
    problem_t p;

    assert(MAXLITERALS <= 8*sizeof(int)-1);
    assert((MAXCLAUSES & (MAXCLAUSES-1)) == 0); /* "bit hack" to check whether MAXCLAUSES is a power of two */

    if (argc != 1) {
        fprintf(stderr, "Usage: %s < input\n", argv[0]);
        return EXIT_FAILURE;
    }
#ifndef SERIAL

    sclInitFromFile("opencl-sat.cl");
    eval_kernel = sclCreateKernel("eval_kernel");
#endif

    load_dimacs(stdin, &p);
    pretty_print(&p);
    const double tstart = hpc_gettime();
    int nsolutions = sat(&p);
    const double elapsed = hpc_gettime() - tstart;
    printf("%d solutions in %f seconds\n", nsolutions, elapsed);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
