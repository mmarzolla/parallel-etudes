/****************************************************************************
 *
 * opencl-sat.cl - OpenCL SAT solver
 *
 * Copyright (C) 2024 Moreno Marzolla
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

#include <stdint.h>

#define MAXLITERALS 30
#define MAXCLAUSES 300

// 1D workgroup, with one work-item for each clause
__kernel void
eval_kernel(__global const int lit[MAXCLAUSES][MAXLITERALS],
            int nlit,
            int nclauses,
            int v,
            __global int *nsat)
{
    __local bool temp[MAXCLAUSES];
    const int lindex = get_local_id(0);
    const int gindex = get_global_id(0);
    const int c = lindex;

    v += gindex;

    temp[c] = c >= nclauses;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (c >= nclauses)
        return;

    for (int l=0; lit[c][l]; l++) {
        int x = lit[c][l];
        if (x > 0) {
            x--;
            temp[c] |= ((v & (1 << x)) != 0);
        } else {
            x++;
            temp[c] |= !((v & (1 << (-x))) != 0);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // Reduce using logical "and" along dimension 1
    for (int bsize = nclauses / 2; bsize > 0; bsize /= 2) {
        if ( c < bsize ) {
            temp[c] &= temp[c + bsize];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (c == 0)
        nsat[gindex] += temp[0];
}
