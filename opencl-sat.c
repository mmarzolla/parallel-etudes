/****************************************************************************
 *
 * opencl-sat.c - Brute-force SAT solver
 *
 * Copyright (C) 2018, 2023, 2024 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Brute-force SAT solver
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2024-09-28

To compile:

        gcc -Wall -Wpedantic opencl-sat.c simpleCL.c -o opencl-sat -LOpenCL

To execute:

        ./opencl-sat < sat.cnf

## Files

- [opencl-sat.c](opencl-sat.c)
- Some input files: <queens-05.cnf>, <uf20-01.cnf>, <uf20-077.cnf>
***/

#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include "simpleCL.h"
#include "hpc.h"

/* MAXLITERALS must be at most (bit width of int) - 2 */
#define MAXLITERALS 30
/* MAXCLAUSES must be a power of two */
#define MAXCLAUSES 512

typedef struct {
    int lit[MAXCLAUSES][MAXLITERALS];
    int nlit;
    int nclauses;
} problem_t;

sclKernel eval_kernel;

int max(int a, int b)
{
    return (a>b ? a : b);
}

int abs(int x)
{
    return (x>=0 ? x : -x);
}

/**
 * OpenCL implementation of a brute-force SAT solver. This
 * implementation uses 1D workgroup of 1D work-items. Each work-item
 * has `nclauses` threads and evaluates a clause. Multiple work-items
 * evaluate multiple clauses in parallel. We can not launch `nclauses`
 * work-items, since that might exceed hardware limits. Therefore,
 * multiple kernel launches are required in the "for" loop below.
 */
int sat( const problem_t *p)
{
    const int NLIT = p->nlit;
    const int NCLAUSES = p->nclauses;
    const int MAX_VALUE = (1 << NLIT) - 1;
    const sclDim BLOCK = DIM1(NCLAUSES);
    const int GRID_SIZE = 1 << 18;
    const sclDim GRID = DIM1(GRID_SIZE * NCLAUSES);
    const int NSAT_SIZE = sizeof(int) * GRID_SIZE;

    int *nsat = (int*)malloc(NSAT_SIZE); assert(nsat);
    cl_mem d_nsat, d_lit;
    int cur_value;

    for (int i=0; i<GRID_SIZE; i++) {
        nsat[i] = 0;
    }
    d_lit = sclMallocCopy(MAXLITERALS * MAXCLAUSES * sizeof(int), (void*)(p->lit), CL_MEM_READ_ONLY);
    d_nsat = sclMallocCopy(NSAT_SIZE, nsat, CL_MEM_READ_WRITE);

    for (cur_value=0; cur_value<=MAX_VALUE; cur_value += GRID_SIZE) {
        sclSetArgsEnqueueKernel(eval_kernel,
                                GRID, BLOCK,
                                ":b :d :d :d :b",
                                d_lit,
                                p->nlit,
                                p->nclauses,
                                cur_value,
                                d_nsat);
    }
    sclMemcpyDeviceToHost(nsat, d_nsat, NSAT_SIZE);

    int result = 0;
    for (int i=0; i<GRID_SIZE; i++)
        result += nsat[i];

    free(nsat);
    sclFree(d_nsat);
    sclFree(d_lit);
    return result;
}

/**
 * Pretty-prints problem |p|
 */
void pretty_print( const problem_t *p )
{
    int c, l;
    for (c=0; (c < MAXCLAUSES) && p->lit[c][0]; c++) {
        printf("( ");
        for (l=0; (l < MAXLITERALS) && p->lit[c][l]; l++) {
            if (p->lit[c][l] > 0 ) {
                printf("x_%d ", p->lit[c][l]);
            } else {
                printf("¬x_%d ", -(p->lit[c][l]));
            }
            if ((l < MAXLITERALS-1) && p->lit[c][l+1] ) {
                printf("∨ ");
            }
        }
        printf(")");
        if ((c < MAXCLAUSES-1) && p->lit[c+1][0]) {
            printf(" ∧");
        }
        printf("\n");
    }
}

/**
 * Load a DIMACS CNF file |f| and initialize problem |p|.  The DIMACS
 * CNF format specification fan be found at
 * https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/satformat.ps
 */
void load_dimacs( FILE *f, problem_t *p )
{
    int result;
    int c, l, val;
    int prob_c, prob_l;

    /* Set all literals to false */
    for (c=0; c<MAXCLAUSES; c++) {
        for (l=0; l<MAXLITERALS; l++) {
            p->lit[c][l] = 0;
        }
    }
    p->nlit = -1;
    c = l = 0;
    /* From
       https://github.com/marijnheule/march-SAT-solver/blob/master/parser.c */
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
            /* New clause */
            l = 0;
            c++;
        } else {
            /* New literal */
            p->lit[c][l] = val;
            p->nlit = max(p->nlit, abs(val));
            l++;
        }
    }
    p->nclauses = c;
    fprintf(stderr, "DIMACS CNF files: %d clauses, %d literals\n", c, p->nlit);
}

int main( void )
{
    problem_t p;

    assert(MAXLITERALS <= 8*sizeof(int)-2);
    assert((MAXCLAUSES & (MAXCLAUSES-1)) == 0); /* "bit hack" to check whether MAXCLAUSES is a power of two */

    sclInitFromFile("opencl-sat.cl");
    eval_kernel = sclCreateKernel("eval_kernel");

    load_dimacs(stdin, &p);
    pretty_print(&p);
    const double tstart = hpc_gettime();
    int nsolutions = sat(&p);
    const double elapsed = hpc_gettime() - tstart;
    printf("%d solutions in %f seconds\n", nsolutions, elapsed);
    sclFinalize();
    return EXIT_SUCCESS;
}