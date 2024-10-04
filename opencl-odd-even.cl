/****************************************************************************
 *
 * opencl-odd-evenl.cl -- kernel for opencl-odd-even.c
 *
 * Copyright (C) 2021 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
