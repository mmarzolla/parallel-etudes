/****************************************************************************
 *
 * opencl-reverse.cl -- Kernels for opencl-reverse.c
 *
 * Copyright (C) 2021 Moreno Marzolla <https://www.moreno.marzolla.name/>
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

/* Reverse in[] into out[]; n work items are required to reverse n
   elements */
__kernel void
reverse_kernel( __global const int *in,
                __global int *out,
                int n )
{
    const int i = get_global_id(0);
    if ( i < n ) {
        const int opp = n - 1 - i;
        out[opp] = in[i];
    }
}

/* In-place reversal of in[]; n/2 work-items are required to reverse
   n elements */
__kernel void
inplace_reverse_kernel( __global int *in,
                        int n )
{
    const int i = get_global_id(0);
    if ( i < n/2 ) {
        const int opp = n - 1 - i;
        const int tmp = in[opp];
        in[opp] = in[i];
        in[i] = tmp;
    }
}
