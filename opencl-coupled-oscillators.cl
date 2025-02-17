/****************************************************************************
 *
 * opencl-coupled-oscillators.cl - Kernel for opencl-coupled-oscillator.c
 *
 * Copyright (C) 2021 Moreno Marzolla <https://www.unibo.it/sitoweb/moreno.marzolla/>
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
step_kernel( __global const float *x,
             __global const float *v,
             __global float *xnext,
             __global float *vnext,
             float k,
             float m,
             float dt,
             int n )
{
    const int i = get_global_id(0);

    if ( i >= n )
        return;

    if ( i > 0 && i < n - 1 ) {
        /* Compute the net force acting on mass i */
        const float F = k*(x[i-1] - 2*x[i] + x[i+1]);
        const float a = F/m;
        /* Compute the next position and velocity of mass i */
        vnext[i] = v[i] + a*dt;
        xnext[i] = x[i] + vnext[i]*dt;
    } else {
        /* First and last values of x and v are just copied to the new arrays; */
        xnext[i] = x[i];
        vnext[i] = 0.0;
    }
}
