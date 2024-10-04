/****************************************************************************
 *
 * opencl-dot-local.cl -- kernel for opencl-dot-local.c
 *
 * Copyright (C) 2021 Moreno Marzolla <moreno.marzolla@unibo.it>
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
dot_kernel( __global const float *x,
            __global const float *y,
            int n,
            __global float *result )
{
    __local float sums[SCL_DEFAULT_WG_SIZE];
    float local_sum = 0.0;

    const int tid = get_local_id(0);
    const int nitems = get_local_size(0);
    int i;

    for (i = tid; i < n; i += nitems) {
        local_sum += x[i] * y[i];
    }
    sums[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE); /* Wait for all work-items to write to the shared array */
    /* Work-item 0 makes the final reduction */
    if ( 0 == tid ) {
        float sum = 0.0;
        for (i=0; i<nitems; i++) {
            sum += sums[i];
        }
        *result = sum;
    }
}
