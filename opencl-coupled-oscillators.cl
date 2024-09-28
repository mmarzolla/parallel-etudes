/****************************************************************************
 *
 * opencl-coupled-oscillators.cl -- Kernel for opencl-coupled-oscillator.c
 *
 * Copyright (C) 2021 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
