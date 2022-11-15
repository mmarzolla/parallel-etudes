/****************************************************************************
 *
 * opencl-bsearch.cl -- kernel for opencl-bsearch.c
 *
 * Copyright 2022 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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

/* Cerca il valore key nell'array x[] di lunghezza n; se presente
  `*result` conterrà l'indice della prima occorrenza, altrimenti -1 */
__kernel void
bsearch_kernel( __global const int *x,
                int n, /* FIXME: dovrebbe essere size_t */
                int key,
                __global int *result,
                __local int *cmp,
                __local size_t *m)
{
    const int bsize = get_local_size(0);
    const int tid = get_global_id(0);
    __local size_t start, end;

    if (0 == tid) {
        start = 0;
        end = n-1;
        *result = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    while (end - start > bsize) {
        m[tid] = start + ((end - start) * tid + bsize) / (bsize+1);

        if (x[m[tid]] < key)
            cmp[tid] = 1;
        else
            cmp[tid] = -1;

        barrier(CLK_LOCAL_MEM_FENCE);

        /* cmp[tid] == 1 -> vai a destra
           cmp[tid] == -1 -> vai a sinistra */

        /* asserzione:
           cmp[i] == 1 -> key in posizione > m[i]
           cmp[i] == -1 -> key in posizione <= m[i] */
        if (tid == 0 && cmp[tid] == -1) {
            end = m[tid];
        } else if (tid == bsize-1 && cmp[tid] == 1) {
            start = m[tid] + 1;
        } else if (tid>0 && cmp[tid-1] == 1 && cmp[tid] == -1) {
            start = m[tid-1] + 1;
            end = m[tid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // final result
    const int idx = start + tid;
    if (idx < end && x[idx] == key) {
        *result = idx;
    }
}
