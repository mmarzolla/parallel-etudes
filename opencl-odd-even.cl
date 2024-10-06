/****************************************************************************
 *
 * opencl-odd-evenl.cl -- kernel for opencl-odd-even.c
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

void cmp_and_swap( __global int* a, __global int* b )
{
    if ( *a > *b ) {
        int tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

/**
 * This kernel requires `n` work-items to sort `n` elements, but only
 * half the work-items are used during each phase. Therefore, this
 * kernel is not efficient.
 */
__kernel void
step_kernel_bad( __global int *x,
                 int n,
                 int phase )
{
    const int idx = get_global_id(0);
    if ( (idx < n-1) && ((idx % 2) == (phase % 2)) ) {
        /* Compare & swap x[idx] and x[idx+1] */
        cmp_and_swap(&x[idx], &x[idx+1]);
    }
}

/**
 * A more efficient kernel that uses n/2 work-items
 */
__kernel void
step_kernel_good( __global int *x,
                  int n,
                  int phase )
{
    const int tid = get_global_id(0); /* thread index */
    const int idx = tid*2 + (phase % 2); /* array index handled by this thread */
    if (idx < n-1) {
        cmp_and_swap(&x[idx], &x[idx+1]);
    }
}
