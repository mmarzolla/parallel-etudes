/****************************************************************************
 *
 * opencl-nbody-simd.cl - SIMD kernels for opencl-nbody.c
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
#define EPSILON 1.0e-5f

__kernel void
compute_force_kernel(__global const float3 *x,
                     __global float3 *v,
                     float dt,
                     int n)
{
    const int i = get_global_id(0);

    if (i<n) {
        float3 F = (float3)(0.0f, 0.0f, 0.0f);

        for (int j = 0; j < n; j++) {
            const float3 dx = x[j] - x[i];
            const float invDist = 1.0f / (distance(x[j], x[i]) + EPSILON);
            const float invDist3 = invDist * invDist * invDist;

            F += dx * invDist3;
        }
        v[i] += dt*F;
    }
}

__kernel void
integrate_positions_kernel(__global float3 *x,
                           __global const float3 *v,
                           float dt,
                           int n)
{
    const int i = get_global_id(0);
    if (i < n) {
        x[i] += v[i]*dt;
    }
}

__kernel void
energy_kernel(__global const float3 *x,
              __global const float3 *v,
              int n,
              __global float *results)
{
    __local float temp[SCL_DEFAULT_WG_SIZE];

    const int gi = get_global_id(0);
    const int li = get_local_id(0);
    const int gid = get_group_id(0);

    temp[li] = 0.0f;

    if (gi < n) {
        temp[li] =  0.5f*dot(v[gi], v[gi]);
        for (int gj= gi+1; gj<n; gj++) {
            temp[li] -= 1.0f / distance(x[gi], x[gj]);
        }
    }
    /* wait for all work-items to finish the copy operation */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* All work-items cooperate to compute the local sum */
    for (int bsize = get_local_size(0)/2; bsize > 0; bsize /= 2) {
        if ( li < bsize ) {
            temp[li] += temp[li + bsize];
        }
        /* threads must synchronize before performing the next
           reduction step */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ( 0 == li ) {
        results[gid] = temp[0];
    }
}
