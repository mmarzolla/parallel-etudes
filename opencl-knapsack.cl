/****************************************************************************
 *
 * opencl-knapsack.cl -- Kernel for opencl-knapsack.c
 *
 * Copyright (C) 2022 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
knapsack_first_row( __global float *P,
                    __global int *w,
                    __global const float *v,
                    int NCOLS )
{
    const int j = get_global_id(0);
    if ( j < NCOLS )
	P[j] = (j < w[0] ? 0.0f : v[0]);
}

__kernel void
knapsack_step( __global const float *Vcur,
               __global float *Vnext,
               __global const int *w,
               __global const float *v,
               int i,
               int NCOLS )
{
    const int j = get_global_id(0);
    if ( j < NCOLS ) {
        if ( j >= w[i] ) {
            Vnext[j] = max(Vcur[j], Vcur[j - w[i]] + v[i]);
        } else {
            Vnext[j] = Vcur[j];
        }
    }
}
