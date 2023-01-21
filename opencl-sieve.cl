/****************************************************************************
 *
 * opencl-sieve.cl -- kernel for opencl-sieve.c
 *
 * Copyright 2023 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
__kernel void
next_prime_kernel(__global const char *isprime,
                  int k,
                  int n,
                  __global int *next_prime)
{
    const int li = get_local_id(0);
    if (0 == li) {
        k++;
        while (k < n && isprime[k] == 0)
            k++;
        *next_prime = k;
    }
}
