/****************************************************************************
 *
 * opencl-nbody.cl - Kernels for opencl-nbody.c
 *
 * Copyright (C) 2021, 2022 Moreno Marzolla <https://www.unibo.it/sitoweb/moreno.marzolla/>
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
compute_force_kernel_local(__global const float *x,
                           __global const float *y,
                           __global const float *z,
                           __global float *vx,
                           __global float *vy,
                           __global float *vz,
                           const float dt,
                           const int n)
{
    __local float tmp_x[SCL_DEFAULT_WG_SIZE];
    __local float tmp_y[SCL_DEFAULT_WG_SIZE];
    __local float tmp_z[SCL_DEFAULT_WG_SIZE];

    const int li = get_local_id(0);
    const int gi = get_global_id(0);
    const int BSIZE = get_local_size(0);
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int b = 0; b < n; b += BSIZE) {

        /* Care should be taken if the number of particles is not
           a multiple of BSIZE */
        const int DIM = min(n - b, BSIZE);

        if (li < DIM) {
            tmp_x[li] = x[b + li];
            tmp_y[li] = y[b + li];
            tmp_z[li] = z[b + li];
        }

        /* Wait for all threads to fill the local memory */
        barrier(CLK_LOCAL_MEM_FENCE);

        if (gi < n) {
            for (int j = 0; j < DIM; j++) {
                const float dx = tmp_x[j] - x[gi];
                const float dy = tmp_y[j] - y[gi];
                const float dz = tmp_z[j] - z[gi];

                const float distSqr = dx*dx + dy*dy + dz*dz + EPSILON;
                const float invDist = 1.0f / sqrt(distSqr);
                const float invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }

        /* Wait for all work-items to finish the computation before
           modifying the local memory at the next iteration */
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (gi < n) {
        vx[gi] += dt*Fx;
        vy[gi] += dt*Fy;
        vz[gi] += dt*Fz;
    }
}

__kernel void
compute_force_kernel(__global const float *x,
                     __global const float *y,
                     __global const float *z,
                     __global float *vx,
                     __global float *vy,
                     __global float *vz,
                     const float dt,
                     const int n)
{
    /* This version does not use local memory */
    const int i = get_global_id(0);

    if (i<n) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            const float dx = x[j] - x[i];
            const float dy = y[j] - y[i];
            const float dz = z[j] - z[i];
            const float distSqr = dx*dx + dy*dy + dz*dz + EPSILON;
            const float invDist = 1.0f / sqrt(distSqr);
            const float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        vx[i] += dt*Fx;
        vy[i] += dt*Fy;
        vz[i] += dt*Fz;
    }
}

__kernel void
integrate_positions_kernel(__global float *x,
                           __global float *y,
                           __global float *z,
                           __global const float *vx,
                           __global const float *vy,
                           __global const float *vz,
                           const float dt,
                           const int n)
{
    const int i = get_global_id(0);
    if (i < n) {
        x[i] += vx[i]*dt;
        y[i] += vy[i]*dt;
        z[i] += vz[i]*dt;
    }
}

__kernel void
energy_kernel(__global const float *x,
              __global const float *y,
              __global const float *z,
              __global const float *vx,
              __global const float *vy,
              __global const float *vz,
              const int n,
              __global float *results)
{
    __local float temp[SCL_DEFAULT_WG_SIZE];

    const int gi = get_global_id(0);
    const int li = get_local_id(0);
    const int gid = get_group_id(0);

    temp[li] = 0.0f;

    if (gi < n) {
        temp[li] = 0.5f * (vx[gi]*vx[gi] + vy[gi]*vy[gi] + vz[gi]*vz[gi]);
        for (int gj=gi+1; gj<n; gj++) {
            const float dx = x[gi] - x[gj];
            const float dy = y[gi] - y[gj];
            const float dz = z[gi] - z[gj];
            const float distance = sqrt(dx*dx + dy*dy + dz*dz);
            temp[li] -= 1.0f / distance;
        }
    }
    /* wait for all work-items to finish the copy operation */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* All work-items cooperate to compute the local sum */
    for (int bsize = get_local_size(0)/2; bsize > 0; bsize /= 2) {
        if ( li < bsize ) {
            temp[li] += temp[li + bsize];
        }
        /* work-items must synchronize before performing the next
           reduction step */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ( 0 == li ) {
        results[gid] = temp[0];
    }
}
