/****************************************************************************
 *
 * opencl-mandelbrot-area.cl - Area of the Mandelbrot set
 *
 * Copyright (C) 2022--2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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

int __constant MAXIT = 10000;

typedef unsigned int uint32_t;

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first `n` such that `z_n > bound`, or `MAXIT` if `z_n` is below
 * `bound` after `MAXIT` iterations.
 */
int iterate( float cx, float cy )
{
    float x = 0.0f, y = 0.0f, xnew, ynew;
    int it;
    for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0f*2.0f); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0f*x*y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

__kernel void
mandelbrot_area_kernel( int xsize,
                        int ysize,
                        __global uint32_t *ninside)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const float XMIN = -2.25, XMAX = 0.75;
    const float YMIN = -1.4, YMAX = 1.5;

    __local uint32_t local_inside[SCL_DEFAULT_WG_SIZE2D][SCL_DEFAULT_WG_SIZE2D];

    if (x < xsize && y < ysize) {
        const float cx = XMIN + (XMAX - XMIN) * x / xsize;
        const float cy = YMIN + (YMAX - YMIN) * y / ysize;
        const int v = iterate(cx, cy);
        local_inside[ly][lx] = (v >= MAXIT);
    } else
        local_inside[ly][lx] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Column-wise reduction */
    for ( int bsize = get_local_size(0) / 2; bsize > 0; bsize /= 2 ) {
        if ( lx < bsize ) {
            local_inside[ly][lx] += local_inside[ly][lx + bsize];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Row-wise reduction on first column */
    if (lx == 0) {
        for ( int bsize = get_local_size(1) / 2; bsize > 0; bsize /= 2 ) {
            if ( ly < bsize ) {
                local_inside[ly][0] += local_inside[ly + bsize][0];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (ly == 0)
            atomic_add(ninside, local_inside[0][0]);
    }
}
