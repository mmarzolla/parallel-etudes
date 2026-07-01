/****************************************************************************
 *
 * opencl-rank.cl - kernel for opencl-rank.c
 *
 * Copyright (C) 2026 Moreno Marzolla
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

__kernel void
rank_kernel( __global const int *v, __global int *rank, int n )
{
    __local int other_v[SCL_DEFAULT_WG_SIZE];
    __local int local_rank[SCL_DEFAULT_WG_SIZE];
    const int li = get_local_id(0);
    const int bsize = get_local_size(0);
    const int gi = get_global_id(0);

    if (gi >= n)
        return;

    local_rank[li] = 0;

    /* Loop over tiles */
    for (int t=0; t<n; t += bsize) {
        const int this_tile_size = (t + bsize <= n ? bsize : n % bsize);
        /* Fetch an element and populate the tile; make sure we don't
           fetch outside the array bound. */
        if (li < this_tile_size)
            other_v[li] = v[t + li];
        barrier(CLK_LOCAL_MEM_FENCE);
        /* compare v[gi] to other_v[]; gj is the global index
           corresponding to the local index lj. */
        for (int lj=0, gj = t; lj<this_tile_size; lj++, gj++) {
            if ( (v[gi] > other_v[lj]) || (v[gi] == other_v[lj] && gi < gj) )
                local_rank[li]++;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Update ranks */
    rank[gi] = local_rank[li];
}

