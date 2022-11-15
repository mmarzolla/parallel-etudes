/****************************************************************************
 *
 * opencl-mandelbrot-area.cl - Area of the Mandelbrot set
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

/* The state word must be initialized to non-zero */
uint32_t xorshift32(uint32_t *state)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

float randab(uint32_t *state, float a, float b)
{
    return a + (b-a)*(xorshift32(state)/(float)UINT_MAX);
}

__kernel void
mandelbrot_area_kernel( uint32_t seed,
                        uint32_t npoints,
                        __global uint32_t *ninside)
{
    uint32_t state = seed + 17 * get_global_id(0);
    /* We consider the region on the complex plane -2.25 <= Re <= 0.75
       -1.4 <= Im <= 1.5 */
    const float XMIN = -2.25, XMAX = 0.75;
    const float YMIN = -1.5, YMAX = 1.5;

    if (get_global_id(0) < npoints) {
        const float cx = randab(&state, XMIN, XMAX);
        const float cy = randab(&state, YMIN, YMAX);
        const int v = iterate(cx, cy);
        if (v >= MAXIT)
            atomic_inc(ninside);
    }
}
