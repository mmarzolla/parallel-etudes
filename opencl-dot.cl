/****************************************************************************
 *
 * opencl-dot.cl - kernel for opencl-dot.c
 *
 * Copyright (C) 2021, 2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
dot_kernel( __global const float *x,
            __global const float *y,
            int n,
            __global float *result )
{
    __local float tmp[SCL_DEFAULT_WG_SIZE];
    float local_sum = 0.0;

    const int tid = get_local_id(0);
    const int nitems = get_local_size(0);

    for (int i = tid; i < n; i += nitems) {
        local_sum += x[i] * y[i];
    }
    tmp[tid] = local_sum;
    /* Wait for all work-items to write to the shared array */
    barrier(CLK_LOCAL_MEM_FENCE);
    /* Work-item 0 makes the final reduction */
    if ( 0 == tid ) {
        float sum = 0.0;
        for (int i=0; i<nitems; i++) {
            sum += tmp[i];
        }
        *result = sum;
    }
}
