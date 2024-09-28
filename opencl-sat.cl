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

#define MAXLITERALS 30
#define MAXCLAUSES 512

// 1D workgroup, with one work-item for each clause
__kernel void
eval_kernel(__global const int lit[MAXCLAUSES][MAXLITERALS],
            int nlit,
            int nclauses,
            int v,
            __global int *nsat)
{
    __local bool term[MAXCLAUSES];
    const int lindex = get_local_id(0);
    const int gindex = get_group_id(0);
    const int c = lindex;
    const int max_value = (1 << nlit) - 1;

    v += gindex;

    if (v > max_value || c >= nclauses)
        return;

    term[c] = false;

    /* In the CNF format, literals are indexed from 1; therefore, the
       bit mask must be shifted left one position. */
    v = v << 1;
    for (int l=0; lit[c][l]; l++) {
        int x = lit[c][l];
        if (x > 0) {
            term[c] |= ((v & (1 << x)) != 0);
        } else {
            term[c] |= !((v & (1 << (-x))) != 0);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // Reduce using logical "and"; we require that `MAXCLAUSES` be a
    // power of two in order for the reduction to work. Actually, a
    // more efficient solution would be to round `nclauses` to the
    // next power of two.
    for (int bsize = MAXCLAUSES / 2; bsize > 0; bsize /= 2) {
        if ( c + bsize < nclauses ) {
            term[c] &= term[c + bsize];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (c == 0)
        nsat[gindex] += term[0];
}
