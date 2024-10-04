/****************************************************************************
 *
 * opencl-letters.cl -- kernel for opencl-letters.c
 *
 * Copyright (C) 2023 Moreno Marzolla <https://www.moreno.marzolla.name/>
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

#define ALPHA_SIZE 26

__kernel
void hist_kernel( __global const char *text,
                  int len,
                  __global int *hist )
{
    const int i = get_global_id(0);
    const int li = get_local_id(0);
    __local int local_hist[ALPHA_SIZE];

    /* reset local histogram */
    if (li < ALPHA_SIZE)
        local_hist[li] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < len) {
        char c = text[i];

        if (c >= 'A' && c <= 'Z')
            c = (c - 'A') + 'a';

        if (c >= 'a' && c <= 'z')
            atomic_inc(&local_hist[ c - 'a' ]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (li < ALPHA_SIZE)
        atomic_add(&hist[li], local_hist[li]);
}
