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

__kernel void
eval_kernel(__global const int *x,
            __global const int *nx,
            int nlit,
            int nclauses,
            int v,
            __global int *nsat)
{
    __local bool exp; // Value of the expression handled by this work-item
    const int lindex = get_local_id(0);
    const int gindex = get_group_id(0);
    const int c = lindex;
    const int MAX_VALUE = (1 << nlit) - 1;

    v += gindex;

    if (v > MAX_VALUE || c >= nclauses)
        return;

    if (c == 0)
        exp = true;

    barrier(CLK_LOCAL_MEM_FENCE);

    const bool term = (v & x[c]) | (~v & nx[c]);

    /* If one term is false, the whole expression is false. */
    if (! term)
        exp = false;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (c == 0)
        nsat[gindex] += exp;
}
