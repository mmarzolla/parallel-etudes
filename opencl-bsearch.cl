/****************************************************************************
 *
 * opencl-bsearch.cl -- kernel for opencl-bsearch.c
 *
 * Copyright (C) 2022--2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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

/* Find the position `*pos` of value `key` into the array `x[]` of
   length `n`. `x[]` must be sorted in nondecreasing order; at most
   one occurrence of `key` must be present. If `key` does not appear
   in `x`, set `*pos` to -1.  `cmp[]` and `m[]` are buffers of length
   `bsize` in local memory. */
__kernel void
bsearch_kernel( __global const int *x,
                int n,
                int key,
                __global int *pos,
                __local int *cmp,
                __local size_t *m)
{
    const int bsize = get_local_size(0);
    const int tid = get_global_id(0);
    __local size_t start, end;

    // Initialization
    if (0 == tid) {
        start = 0;
        end = n-1;
        *pos = -1;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    while (end - start > bsize) {
        m[tid] = start + ((end - start) * tid + bsize) / (bsize+1);

        if (x[m[tid]] < key)
            cmp[tid] = 1;
        else
            cmp[tid] = -1;

        barrier(CLK_LOCAL_MEM_FENCE);

        /* cmp[tid] == 1 -> key is on the right
           cmp[tid] == -1 -> key is on the left */

        /* assertion:

           cmp[i] == 1 -> if key exists, it is in a position > m[i]
           cmp[i] == -1 -> if key exists, it is in a position <= m[i]

           If there is at most one occurrence of `key` in `x[]`, then
           only one work-item updates `end` and `start`, so no
           concurrent updates are possible.
        */
        if (tid == 0 && cmp[tid] == -1) {
            end = m[tid];
        } else if (tid == bsize-1 && cmp[tid] == 1) {
            start = m[tid] + 1;
        } else if (tid>0 && cmp[tid-1] == 1 && cmp[tid] == -1) {
            start = m[tid-1] + 1;
            end = m[tid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Commit the result to `*pos`
    if (0 == tid) {
        const int idx = start + tid;
        if (idx < end && x[idx] == key) {
            *pos = idx;
        }
    }
}
