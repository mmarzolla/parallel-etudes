/****************************************************************************
 *
 * opencl-nbody-simd.cl - SIMD kernels for opencl-nbody.c
 *
 * Copyright (C) 2021 Moreno Marzolla
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
