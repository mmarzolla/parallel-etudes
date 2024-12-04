/****************************************************************************
 *
 * opencl-knapsack.cl - Kernel for opencl-knapsack.c
 *
 * Copyright (C) 2022 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
