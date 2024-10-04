/****************************************************************************
 *
 * opencl-cat-map.cl - Kernel per opencl-cat-map.c
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

/**
 * Compute one iteration of the cat map using the GPU
 */
__kernel
void cat_map_iter_kernel( __global const unsigned char *cur,
                          __global unsigned char *next,
                          int N )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ( x < N && y < N ) {
        const int xnext = (2*x+y) % N;
        const int ynext = (x + y) % N;
        next[xnext + ynext*N] = cur[x+y*N];
    }
}

/**
 * Compute `k` iterations of the cat map using the GPU
 */
__kernel
void cat_map_iter_k_kernel( __global const unsigned char *cur,
                            __global unsigned char *next,
                            int N,
                            int k )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ( x < N && y < N ) {
        int xcur = x, ycur = y, xnext, ynext;
        while (k--) {
            xnext = (2*xcur+ycur) % N;
            ynext = (xcur + ycur) % N;
            xcur = xnext;
            ycur = ynext;
        }
        next[xnext + ynext*N] = cur[x+y*N];
    }
}
