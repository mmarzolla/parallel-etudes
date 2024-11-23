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

/* Each work-item checks an assignment */
__kernel
void eval_kernel(__global const int *x,
                 __global const int *nx,
                 int nlit,
                 int nclauses,
                 int v,
                 __global int *nsat)
{
    __local int nsol[SCL_DEFAULT_WG_SIZE];
    const int lindex = get_local_id(0);
    const int gindex = get_global_id(0);
    const int MAX_VALUE = (1 << nlit) - 1;

    v += gindex;

    if (v <= MAX_VALUE) {
        bool result = true;
        for (int c=0; c < nclauses && result; c++) {
            const bool term = (v & x[c]) | (~v & nx[c]);
            result &= term;
        }
        nsol[lindex] = result;
    } else
        nsol[lindex] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // perform a reduction
    for (int bsize = get_local_size(0) / 2; bsize > 0; bsize /= 2) {
        if ( lindex < bsize ) {
            nsol[lindex] += nsol[lindex + bsize];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ( 0 == lindex ) {
        atomic_add(nsat, nsol[0]);
    }
}
