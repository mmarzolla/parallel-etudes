/****************************************************************************
 *
 * opencl-nbody.c -- N-body simulation
 *
 * Copyright (C) Mark Harris
 * Copyright (C) 2021--2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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

/***
% HPC - N-body simulation
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-01-04

![A frame of the Bolshoi simulation (source: <http://hipacc.ucsc.edu/Bolshoi/Images.html>)](bolshoi.png)

In the first lecture at the beginning of the course we have seen a
[video](https://www.youtube.com/watch?v=UngV0zMDEQI) of the [Bolshoi
simulation](http://hipacc.ucsc.edu/Bolshoi.html). Cosmological
simulations study the large-scale evolution of the universe, and are
based on the computation of the dynamics of $N$ masses subject to
mutual gravitational attraction ($N$-body problem). The Bolshoi
simulation required 6 million CPU hours on one of the most powerful
supercomputers of the time. In this exercise we solve the problem for
a small number $N$ of bodies using a very simple algorithm, based on a
developed program by Mark Harris available at
<https://github.com/harrism/mini-nbody> (the program proposed in this
exercise is a modified version of the original).

The physical laws governing the dynamics of $N$ masses were discovered
by [Isaac Newton](https://en.wikipedia.org/wiki/Isaac_Newton): this is
the [second law of
dynamic](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion#Newton's_second_law)
and the [law of universal
gravitation](https://en.wikipedia.org/wiki/Newton's_law_of_universal_gravitation). The
second law of dynamics states that a force $\textbf{F}$ that acts on a
particle of mass $m$ produces an acceleration $\textbf{a}$ such that
$\textbf{F}=m\textbf{a}$. The law of universal gravitation states that
two masses $m_1$ and $m_2$ at a distance $d$ are subject to an
attractive force of magnitude $F = G m_1 m_2 / d^2$ where $G$ is the
gravitational constant ($G \approx 6,674 \times 10^{-11}\ \mathrm{N}\
\mathrm{m}^2\ \mathrm{kg}^{-2}$).

The following explanation is not essential for solving this exercise,
but might be informative and only requires basic physics knowledge.

Let us consider $N$ point-like masses $m_0, \ldots, m_{n-1}$ that are
subject to mutual gravitational attraction only. Since the masses are
point-like, they never collide with each other. Let $\textbf{x}_i: =
(x_i, y_i, z_i)$ be the position and $\textbf{v}_i : = (vx_i, vy_i,
vz_i)$ the velocity vector of mass $i$ at a given time $t$. To compute
the new positions $\textbf{x}'_i$ at time $t': = t + \Delta t$ we
proceed as follows:

1. Compute the total force $\textbf{F}_i$ acting on mass $i$ at time $t$:
$$
\textbf{F}_i: = \sum_{i \neq j} \frac{G m_i m_j} {d_{ij}^2} \textbf {n}_{ij}
$$
where $G$ is the gravitational constant, $d_{ij}^2$ is the
square of the distance between particles $i$ and $j$, and
$\textbf{n}_{ij}$ is the unit vector from particle $i$ to particle $j$

2. Compute the acceleration $\textbf{a}_i$ acting on mass $i$:
$$
\textbf{a}_i: = \textbf{F}_i / m_i
$$

3. Compute the _new_ velocity $\textbf{v}'_i$ of mass $i$ at time $t'$:
$$
\textbf{v}'_i: = \textbf{v}_i + \textbf{a}_i \Delta t
$$

4. Compute the _new_ position $\textbf{x}'_i$ of mass $i$ at time $t'$:
$$
\textbf{x}'_i: = \textbf{x}_i + \textbf{v}'_i \Delta t
$$

The previous steps solve the equations of motion using
Euler's scheme [^1].

[^1]: Euler's integration is numerically unstable on this problem, and
      therefore more accurate but more comples schemes are used in
      practice.

In this program we trade accuracy in favor of simplicity by ignoring
the factor $G m_i m_j$ and rewriting the sum as:

$$
\textbf{F} _i: = \sum_{j = 0}^N \frac{\textbf{d}_{ij}}{(d_{ij}^2 + \epsilon)^{3/2}}
$$

where $\textbf{d}_{ij}$ is the vector from particle $i$ to particle
$j$, i.e., $\textbf{d}_{ij} := (\textbf{x}_j - \textbf{x}_i)$, and the
term $\epsilon > 0$ is used to avoid a division by zero when $i = j$,
that is, when computing the interaction of a particle with itself.

***/

/***
Modify the provided program to use the GPU. A first version can be
easily obtained from the serial version.

Then, observe that the data of each particle is (re)read from device
memory $N$ times; indeed, each of the $N$ work-items scans the entire
`p[]` array, so each element of `p[]` is accessed $N$ times by $N$
different work-items.

![Figure 1: Using _local memory_](opencl-nbody.svg)

In situations like this, it can be useful to try to use use local
memory to reduce access to the device memory (Figure 1). To do this:

a. Each workgroup copies a block of elements from `p[]` in global
   memory in an array `tmp[]` in local memory;

b. Each work-item computes the total force acting a particle
   $p_i$. Hence, work-item $i$ compute the interaction between $p_i$
   and all particles in `tmp[]`;

c. Once the previous step is finished, the program copies the next
   block of particles from global memory to the `tmp[]` array

d. Each work-item computes the force exerted on particle $p_i$ by the
   new particles in `tmp[]`.

The previous process is repeated until all the particles in `p[]` have
been taken into account. In this way, each element of `p[]` is reread
multiple times (once for each workgroup), instead of $N$ times. Since
the number of workgroups is less than $N$, this reduces the pressure
on the memory of the device, improving the performance of the program.

To compile without using local memory:

        cc opencl-nbody.c simpleCL.c -o opencl-nbody -lm -lOpenCL

To compile using local memory:

        cc -DUSE_LOCAL opencl-nbody.c simpleCL.c -o opencl-nbody-local -lm -lOpenCL

To execute:

        ./opencl-nbody [nsteps [nparticles]]

Example:

        ./opencl-nbody 50 10000

## File

- [opencl-nbody.c](opencl-nbody.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

***/

/* The following #define is required by the implementation of
   hpc_gettime(). It MUST be defined before including any other
   file. */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
#include "hpc.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for sqrtf() */
#include <assert.h>

#include "simpleCL.h"

const float EPSILON = 1.0e-5f;
/* const float G = 6.67e-11; */

/**
 * Return a random value in the range [a, b]
 */
float randab( float a, float b )
{
    return a + (rand() / (float)RAND_MAX) * (b-a);
}

/**
 * Randomly initialize positions and velocities of the `n` particles
 * stored in `p`.
 */
void init(float *x, float *y, float *z,
          float *vx, float *vy, float *vz,
          int n)
{
    for (int i = 0; i < n; i++) {
        x[i] = randab(-1, 1);
        y[i] = randab(-1, 1);
        z[i] = randab(-1, 1);
        vx[i] = randab(-1, 1);
        vy[i] = randab(-1, 1);
        vz[i] = randab(-1, 1);
    }
}

/**
 * Compute the new velocities of all particles in `p`
 */
#ifdef SERIAL
void compute_force(const float *x, const float *y, const float *z,
                   float *vx, float *vy, float *vz,
                   float dt, int n)
{
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            const float dx = x[j] - x[i];
            const float dy = y[j] - y[i];
            const float dz = z[j] - z[i];
            const float distSqr = dx*dx + dy*dy + dz*dz + EPSILON;
            const float invDist = 1.0f / sqrtff(distSqr);
            const float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        vx[i] += dt*Fx;
        vy[i] += dt*Fy;
        vz[i] += dt*Fz;
    }
}

/**
 * Update the positions of all particles in p using the updated
 * velocities.
 */
void integrate_positions(float *x, float *y, float *z,
                         const float *vx, const float *vy, const float *vz,
                         float dt, int n)
{
    for (int i = 0 ; i < n; i++) {
        x[i] += vx[i]*dt;
        y[i] += vy[i]*dt;
        z[i] += vz[i]*dt;
    }
}

#endif
/**
 * Compute the total energy of the system as the sum of kinetic and
 * potential energy.
 */
float energy(const float *x, const float *y, const float *z,
             const float *vx, const float *vy, const float *vz,
             int n)
{
    float e = 0.0f;
    /* The kinetic energy of an n-body system is:

       K = (1/2) * sum_i [m_i * (vx_i^2 + vy_i^2 + vz_i^2)]

    */

    for (int i=0; i<n; i++) {
        e += 0.5f * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        /* Accumulate potential energy, defined as

           sum_{i<j} - m[j] * m[j] / d_ij

         */
        for (int j=i+1; j<n; j++) {
            const float dx = x[i] - x[j];
            const float dy = y[i] - y[j];
            const float dz = z[i] - z[j];
            const float distance = sqrtf(dx*dx + dy*dy + dz*dz);
            e -= 1.0f / distance;
        }
    }
    return e;
}

int main(int argc, char* argv[])
{
    int N = 10000;
    const int MAXBODIES = 50000;
    int nsteps = 10;
    const float DT = 1e-6f; /* time step */
    float *x, *y, *z, *vx, *vy, *vz, en;
#ifndef SERIAL
    cl_mem d_x, d_y, d_z, d_vx, d_vy, d_vz, d_energies;
#endif

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [nbodies [niter]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N > MAXBODIES) {
        fprintf(stderr, "FATAL: too many bodies\n");
        return EXIT_FAILURE;
    }

    if (argc > 2) {
        nsteps = atoi(argv[2]);
    }

    srand(1234);

    printf("%d particles, %d steps\n", N, nsteps);
#ifndef SERIAL
#ifdef USE_LOCAL
    printf("using local memory\n");
#else
    printf("NOT using local memory\n");
#endif
#endif

    const size_t size = N*sizeof(*x);
    x = (float*)malloc(size); assert(x != NULL);
    y = (float*)malloc(size); assert(y != NULL);
    z = (float*)malloc(size); assert(z != NULL);
    vx = (float*)malloc(size); assert(vx != NULL);
    vy = (float*)malloc(size); assert(vy != NULL);
    vz = (float*)malloc(size); assert(vz != NULL);

    init(x, y, z, vx, vy, vz, N); /* Init pos / vel data */

#ifndef SERIAL
    sclInitFromFile("opencl-nbody.cl");
#ifdef USE_LOCAL
    sclKernel compute_force_kernel = sclCreateKernel("compute_force_kernel_local");
#else
    sclKernel compute_force_kernel = sclCreateKernel("compute_force_kernel");
#endif
    sclKernel integrate_positions_kernel = sclCreateKernel("integrate_positions_kernel");
    sclKernel energy_kernel = sclCreateKernel("energy_kernel");
#endif

    const sclDim BLOCK = DIM1(SCL_DEFAULT_WG_SIZE);
    const sclDim GRID = DIM1(sclRoundUp(N, SCL_DEFAULT_WG_SIZE));
    const size_t N_OF_BLOCKS = GRID.sizes[0] / BLOCK.sizes[0];

    d_x = sclMallocCopy(size, x, CL_MEM_READ_WRITE);
    d_y = sclMallocCopy(size, y, CL_MEM_READ_WRITE);
    d_z = sclMallocCopy(size, z, CL_MEM_READ_WRITE);
    d_vx = sclMallocCopy(size, vx, CL_MEM_READ_WRITE);
    d_vy = sclMallocCopy(size, vy, CL_MEM_READ_WRITE);
    d_vz = sclMallocCopy(size, vz, CL_MEM_READ_WRITE);

    /* There are problems if you define the energies[]
       array as:

       float energies[N_OF_BLOCKS];

       Indeed, the program seems to work on the GPU, but not non the
       CPU (energies are computed as NaNs). The problem might be that
       N_OF_BLOCKS might be too large for variable-length arrays.
    */
    float *energies;
    const size_t size_energies = N_OF_BLOCKS * sizeof(*energies);
    energies = (float*)malloc(size_energies); assert(energies != NULL);
    d_energies = sclMalloc(size_energies, CL_MEM_WRITE_ONLY);

    double total_time = 0.0;
    for (int step = 1; step <= nsteps; step++) {
        const double tstart = hpc_gettime();
#ifdef SERIAL
        compute_force(x, y, z, vx, vy, vz, DT, N);
        integrate_positions(x, y, z, vx, vy, vz, DT, N);
        en = energy(x, y, z, vx, vy, vz, N);
#else
        sclSetArgsEnqueueKernel(compute_force_kernel,
                                GRID, BLOCK,
                                ":b :b :b :b :b :b :f :d",
                                d_x, d_y, d_z, d_vx, d_vy, d_vz, DT, N);
        sclSetArgsEnqueueKernel(integrate_positions_kernel,
                                GRID, BLOCK,
                                ":b :b :b :b :b :b :f :d",
                                d_x, d_y, d_z, d_vx, d_vy, d_vz, DT, N);
        sclSetArgsEnqueueKernel(energy_kernel,
                                GRID, BLOCK,
                                ":b :b :b :b :b :b :d :b",
                                d_x, d_y, d_z,
                                d_vx, d_vy, d_vz,
                                N, d_energies);
        sclMemcpyDeviceToHost(energies, d_energies, size_energies);
        en = 0.0f;
        for (int i=0; i<N_OF_BLOCKS; i++) {
            en += energies[i];
        }
#endif
        const double elapsed = hpc_gettime() - tstart;
        total_time += elapsed;

        printf("Iteration %3d/%3d : energy=%f, %.3f seconds\n", step, nsteps, en, elapsed);
        fflush(stdout);
    }
    const double avg_time = total_time / nsteps;

    printf("Average %0.3f Billion Interactions / second\n", 1e-9 * N * N / avg_time);

    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
#ifndef SERIAL
    free(energies);
    sclFree(d_x);
    sclFree(d_y);
    sclFree(d_z);
    sclFree(d_vx);
    sclFree(d_vy);
    sclFree(d_vz);
    sclFree(d_energies);
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
