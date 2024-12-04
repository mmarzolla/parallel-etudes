/****************************************************************************
 *
 * opencl-sieve.cl - kernel for opencl-sieve.c
 *
 * Copyright (C) 2023 Moreno Marzolla <https://www.moreno.marzolla.name/>
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

/**
 * Mark all multiples of k belonging to the set {from, ... to-1}.
 * from must be a multiple of k. The number of elements that are
 * marked for the first time is atomically subtracted from *nprimes.
 */
__kernel void
mark_kernel( __global char *isprime,
             int k,
             int from,
             int to,
             __global int *nprimes,
             __local int *mark)
{
    const int i = from + get_global_id(0)*k;
    const int li = get_local_id(0);

    mark[li] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < to) {
        mark[li] = (isprime[i] == 1);
        isprime[i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);

    int d = get_local_size(0);
    while (d > 1) {
        int d2 = (d + 1)/2;
        if (li + d2 < d) mark[li] += mark[li + d2];
        d = d2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (0 == li) {
        /*
        int count = 0;
        for (int i=0; i < SCL_DEFAULT_WG_SIZE1D; i++) {
            count += mark[i];
        }
        atomic_sub(nprimes, count);
        */
        atomic_sub(nprimes, mark[0]);
    }
}

/**
 * Store in *next_prime the next prime strictly greater than k, or n
 * if we reached the end of array isprime[]
 */
__kernel void __attribute__((reqd_work_group_size(1, 1, 1)))
next_prime_kernel(__global const char *isprime,
                  int k,
                  int n,
                  __global int *next_prime)
{
    k++;
    while (k < n && isprime[k] == 0)
        k++;
    *next_prime = k;
}

/**
 * The same, but using a reduction; this is likely less efficient than
 * the kernel above, since the next prims is unlikely to be too far
 * away from `k`.
 */
__kernel void __attribute__((reqd_work_group_size(SCL_DEFAULT_WG_SIZE1D, 1, 1)))
next_prime_kernel_reduce(__global const char *isprime,
                         int k,
                         int n,
                         __global int *next_prime)
{
    __local int first[SCL_DEFAULT_WG_SIZE1D];
    const int li = get_local_id(0);
    const int vec_size = n - (k+1);
    const int istart = (k+1) + (((long)vec_size) * li) / get_local_size(0);
    const int iend = (k+1) + (((long)vec_size) * (li + 1)) / get_local_size(0);
    int i;
    for (i=istart; i<iend && !isprime[i]; i++)
        ;
    first[li] = (i < iend) ? i : n;
    barrier(CLK_LOCAL_MEM_FENCE);
    /* perform reduction */
    int bsize = get_local_size(0) / 2;
    while (bsize > 0) {
        if (li < bsize ) {
            first[li] = min(first[li], first[li+bsize]);
        }
        bsize /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (0 == li) {
        *next_prime = first[0];
    }
}
