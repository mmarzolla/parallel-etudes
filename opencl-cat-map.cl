/****************************************************************************
 *
 * opencl-cat-map.cl - Kernel per opencl-cat-map.c
 *
 * Copyright (C) 2021 Moreno Marzolla
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
