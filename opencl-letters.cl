/****************************************************************************
 *
 * opencl-letters.cl - kernel for opencl-letters.c
 *
 * Copyright (C) 2023 Moreno Marzolla <https://www.unibo.it/sitoweb/moreno.marzolla/>
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

#define ALPHA_SIZE 26

__kernel
void hist_kernel( __global const char *text,
                  int len,
                  __global int *hist )
{
    const int i = get_global_id(0);
    const int li = get_local_id(0);
    __local int local_hist[ALPHA_SIZE];

    /* reset local histogram */
    if (li < ALPHA_SIZE)
        local_hist[li] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < len) {
        char c = text[i];

        if (c >= 'A' && c <= 'Z')
            c = (c - 'A') + 'a';

        if (c >= 'a' && c <= 'z')
            atomic_inc(&local_hist[ c - 'a' ]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (li < ALPHA_SIZE)
        atomic_add(&hist[li], local_hist[li]);
}
