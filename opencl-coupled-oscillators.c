/******************************************************************************
 *
 * opencl-coupled-oscillators.c - One-dimensional coupled oscillators system
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 ******************************************************************************/

/***
% HPC - One-dimensional coupled oscillators
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2023-06-08

![](coupled_metronomes.jpg)

Let us consider $n$ points of mass $m$ arranged along a straight line
at coordinates $x_0, x_1, \ldots, x_{n-1}$. Adjacent masses are
connected by a spring with elastic constant $k$ and rest length
$L$. The first and last points (those in position $x_0$ and $x_{n-1}$
occupy a fixed position and cannot move.

![Figur3 1: Coupled oscillators](opencl-coupled-oscillators.svg)

Initially, one of the springs is displaced so that a wave of
oscillations is triggered; due to the lack of friction, such
oscillations will go on indefinitely. Using Newton's second law of
motion $F = ma$ and Hooke's law which states that a spring with
elastic parameter $k$ that is compressed by $\Delta x$ exerts a force
$k \Delta x$, we develop a program that, given the initial positions
and velocities, computes the positions and speeds of all masses at any
time $t > 0$. The program is based on an iterative algorithm that,
from positions and speeds of the masses at time $t$, determine the new
positions and velocities at time $t + \Delta t$. In particular, the
function

```C
step(double *x, double *v, double *xnext, double *vnext, int n)
```

computes the new position `xnext[i]` and velocity `vnext[i]` of mass
$i$ at time $t + \Delta t$, $0 \le i < n$, given the current position
`x[i]` and velocity `v[i]` at time $t$.

1. For each $i = 1, \ldots, n-2$, the force $F_i$ acting on mass $i$
   is $F_i := k \times (x_{i-1} -2x_i + x_{i+1})$; note that the force
   does not depend on the length $L$ of the spring at rest. Masses 0
   and $n-1$ are stationary, therefore the forces acting on them are
   not computed.

2. For each $i = 1, \ldots, n-2$ the new velocity $v'_i$ of mass $i$
   at time $t + \Delta t$ is $v'_i := v_i + (F_i / m) \Delta
   t$. Again, masses 0 and $n-1$ are statioary, therefore their
   velocities are always zero.

3. For each $i = 1, \ldots, n-2$ the new position $x'_i$ of mass $i$
   at time $t + \Delta t$ is $x'_i := x_i + v'_i \Delta t$. Masses 0
   and $n-1$ are stationary, therefore their positions at time $t +
   \Delta t$ are the same as those at time $t$: $x'_0 := x_0$,
   $x'_{n-1} := x_{n-1}$.

The file [opencl-coupled-oscillators.c](opencl-coupled-oscillators.c)
contains a serial program that computes the evolution of $n$ coupled
oscillators. The program produces a two-dimensional image
`coupled-oscillators.ppm` where each line shows the potential energies
of the springs at any time (Figure 2).

![Figure 2: potential energy of the springs](coupled-oscillators.svg)

Your task is to parallelize function `step()` by defining additional
OpenCL kernel(s).

To compile:

        cc -std=c99 -Wall -Wpedantic opencl-coupled-oscillators.c simpleCL.c -o opencl-coupled-oscillators -lm -lOpenCL

To execute:

        ./opencl-coupled-oscillators [N]

Example:

        ./opencl-coupled-oscillators 1024

## Files

- [opencl-coupled-oscillators.c](opencl-coupled-oscillators.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h)

 ***/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

/* Number of initial steps to skip, before starting to take pictures */
#define TRANSIENT 50000
/* Number of steps to record in the picture */
#define NSTEPS 800

/* Some physical constants; note that these are defined as symbolic
   values rather than constants, since they must be visible inside a
   kernel functions (and normal constants are not, unless they are
   stored in constant memory on the device) */
/* Integration time step */
#define dt 0.02f
/* spring constant (large k = stiff spring, small k = soft spring) */
#define k 0.2f
/* mass */
#define m 1.0f
/* Length of each spring at rest */
#define L 1.0f

/* Initial conditions: all masses are evenly placed so that the
   springs are at rest; some of the masses are displaced to start the
   movement. */
void init( float *x, float *v, int n )
{
    for (int i=0; i<n; i++) {
        x[i] = i*L;
        v[i] = 0.0;
    }
    /* displace some of the masses */
    x[n/3  ] -= 0.5*L;
    x[n/2  ] += 0.7*L;
    x[2*n/3] -= 0.7*L;
}

/**
 * Performs one simulation step: starting from the current positions
 * `x[]` and velocities `v[]` of the masses, compute the next
 * positions `xnext[]` and velocities `vnext[]`.
 */
#ifdef SERIAL
void step( float *x, float *v, float *xnext, float *vnext, int n )
{
    for (int i=0; i<n; i++) {
        if ( i > 0 && i < n - 1 ) {
            /* Compute the net force acting on mass i */
            const float F = k*(x[i-1] - 2*x[i] + x[i+1]);
            const float a = F/m;
            /* Compute the next position and velocity of mass i */
            vnext[i] = v[i] + a*dt;
            xnext[i] = x[i] + vnext[i]*dt;
        } else {
            xnext[i] = x[i];
            vnext[i] = 0.0;
        }
    }
}
#endif

/**
 * Compute x*x
 */
float squared(float x)
{
    return x*x;
}

/**
 * Compute the maximum energy among all springs.
 */
float maxenergy(const float *x, int n)
{
    float maxenergy = -INFINITY;
    for (int i=1; i<n; i++) {
        maxenergy = fmaxf(0.5*k*squared(x[i]-x[i-1]-L), maxenergy);
    }
    return maxenergy;
}

void dumpenergy(FILE *fout, const float *x, int n, float maxen)
{
    /* Dump spring energies (light color = high energy) */
    maxen = maxenergy(x, n);
    for (int i=1; i<n; i++) {
        const float displ = x[i] - x[i-1] - L;
        const float energy = 0.5*k*squared(displ);
        const float v = fminf(energy/maxen, 1.0);
        fprintf(fout, "%c%c%c", 0, (int)(255*v*(displ<0)), (int)(255*v*(displ>0)));
    }
}

int main( int argc, char *argv[] )
{
    int cur = 0, next;
    float maxen;
    int N = 1024;
    const char* fname = "coupled-oscillators.ppm";
#ifdef SERIAL
    float *x[2], *v[2];
#else
    float *x, *v;
    cl_mem d_x[2], d_v[2];
#endif

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [N]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (2 == argc) {
        N = atoi(argv[1]);
    }

    const size_t size = N * sizeof(float);

    FILE *fout = fopen(fname, "w");
    if (NULL == fout) {
        printf("Cannot open %s for writing\n", fname);
        return EXIT_FAILURE;
    }

    /* Write the header of the output file */
    fprintf(fout, "P6\n");
    fprintf(fout, "%d %d\n", N-1, NSTEPS);
    fprintf(fout, "255\n");

#ifdef SERIAL
    x[0] = (float*)malloc(size); assert(x[0]);
    x[1] = (float*)malloc(size); assert(x[1]);
    v[0] = (float*)malloc(size); assert(v[0]);
    v[1] = (float*)malloc(size); assert(v[1]);
    init(x[cur], v[cur], N);
#else
    sclInitFromFile("opencl-coupled-oscillators.cl");
    sclKernel step_kernel = sclCreateKernel("step_kernel");
    sclDim block, grid;
    sclWGSetup1D(N, &grid, &block);

    /* Allocate host copies of x and v */
    x = (float*)malloc(size); assert(x);
    v = (float*)malloc(size); assert(v);
    init(x, v, N);

    /* Allocate device copies of x and v */
    d_x[cur] = sclMallocCopy(size, x, CL_MEM_READ_WRITE);
    d_x[1-cur] = sclMalloc(size, CL_MEM_READ_WRITE);
    d_v[cur] = sclMallocCopy(size, v, CL_MEM_READ_WRITE);
    d_v[1-cur] = sclMalloc(size, CL_MEM_READ_WRITE);
#endif

    /* Write NSTEPS rows in the output image */
    for (int s=0; s<TRANSIENT + NSTEPS; s++) {
        next = 1 - cur;
#ifdef SERIAL
        step(x[cur], v[cur], x[next], v[next], N);
#else
        sclSetArgsEnqueueKernel(step_kernel,
                                grid, block,
                                ":b :b :b :b :f :f :f :d",
                                d_x[cur], d_v[cur], d_x[next], d_v[next], k, m, dt, N);
#endif
        if (s >= TRANSIENT) {
#ifdef SERIAL
            if (s == TRANSIENT) {
                maxen = maxenergy(x[next], N);
            }
            dumpenergy(fout, x[next], N, maxen);
#else
            sclMemcpyDeviceToHost(x, d_x[next], size);
            if (s == TRANSIENT) {
                maxen = maxenergy(x, N);
            }
            dumpenergy(fout, x, N, maxen);
#endif
        }
        cur = 1 - cur;
    }

#ifdef SERIAL
    free(x[0]);
    free(x[1]);
    free(v[0]);
    free(v[1]);
#else
    free(x);
    free(v);
    sclFree(d_x[0]);
    sclFree(d_x[1]);
    sclFree(d_v[0]);
    sclFree(d_v[1]);
    sclFinalize();
#endif

    fclose(fout);
    return EXIT_SUCCESS;
}
