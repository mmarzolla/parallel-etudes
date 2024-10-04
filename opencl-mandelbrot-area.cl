/****************************************************************************
 *
 * opencl-mandelbrot-area.cl - Area of the Mandelbrot set
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

    __local uint32_t local_inside;

    if (lx == 0 && ly == 0)
        local_inside = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < xsize && y < ysize) {
        const float cx = XMIN + (XMAX - XMIN) * x / xsize;
        const float cy = YMIN + (YMAX - YMIN) * y / ysize;
        const int v = iterate(cx, cy);
        if (v >= MAXIT)
            atomic_inc(&local_inside);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lx == 0 && ly == 0)
        atomic_add(ninside, local_inside);
}
