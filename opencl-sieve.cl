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

__kernel void
mark_kernel( __global char *isprime,
             int k,
             int from,
             int to,
             __global int *nprimes )
{
    const int i = from + get_global_id(0)*k;
    const int li = get_local_id(0);

    __local int mark[SCL_DEFAULT_WG_SIZE1D];

    if (li < SCL_DEFAULT_WG_SIZE1D)
        mark[li] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < to) {
        mark[li] = (isprime[i] == 1);
        isprime[i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* TODO: inefficient reduction */
    if (0 == li) {
        int count = 0;
        for (int i=0; i < SCL_DEFAULT_WG_SIZE1D; i++) {
            count += mark[i];
        }
        atomic_sub(nprimes, count);
    }
}

__kernel void
next_prime_kernel(__global const char *isprime,
                  int k,
                  int n,
                  __global int *next_prime)
{
    const int li = get_local_id(0);
    if (0 == li) {
        while (k < n && isprime[k] == 0)
            k++;
        *next_prime = k;
    }
}
