/****************************************************************************
 *
 * cuda-nbody.cu - N-body simulation
 *
 * Copyright (C) Mark Harris
 * Modified in 2020--2024 Moreno Marzolla
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
% HPC - N-Body simulation
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-01-04

![A frame of the Bolshoi simulation (source: <http://hipacc.ucsc.edu/Bolshoi/Images.html>)](bolshoi.png)

Cosmological simulations study the large-scale evolution of the
universe, and are based on the computation of the dynamics of $N$
masses subject to mutual gravitational attraction ($N$-body
problem). The [Bolshoi
simulation](http://hipacc.ucsc.edu/Bolshoi.html) required 6 million
CPU hours on one of the most powerful supercomputers of the time. In
this exercise we solve the problem for a small number $N$ of bodies
using a very simple algorithm, based on a developed program by Mark
Harris available at <https://github.com/harrism/mini-nbody> (the
program proposed in this exercise is a modified version of the
original).

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

The following explanation is not required for solving this exercise,
but might be informative and only requires basic physics knowledge.

Let us consider $N$ point-like masses $m_0, \ldots, m_{n-1}$ that are
subject to mutual gravitational attraction only. Since the masses are
point-like, they never collide with each other. Let $\textbf{x}_i =
(x_i, y_i, z_i)$ be the position and $\textbf{v}_i = (vx_i, vy_i,
vz_i)$ the velocity vector of mass $i$ at a given time $t$. To compute
the new positions $\textbf{x}'_i$ at time $t' = t + \Delta t$ we
proceed as follows:

1. Compute the total force $\textbf{F}_i$ acting on mass $i$ at time $t$:
$$
\textbf{F}_i := \sum_{i \neq j} \frac{G m_i m_j} {d_{ij}^2} \textbf {n}_{ij}
$$
where $G$ is the gravitational constant, $d_{ij}^2$ is the
square of the distance between particles $i$ and $j$, and
$\textbf{n}_{ij}$ is the unit vector from particle $i$ to particle $j$

2. Compute the acceleration $\textbf{a}_i$ acting on mass $i$:
$$
\textbf{a}_i := \textbf{F}_i / m_i
$$

3. Compute the _new_ velocity $\textbf{v}'_i$ of mass $i$ at time $t'$:
$$
\textbf{v}'_i := \textbf{v}_i + \textbf{a}_i \Delta t
$$

4. Compute the _new_ position $\textbf{x}'_i$ of mass $i$ at time $t'$:
$$
\textbf{x}'_i := \textbf{x}_i + \textbf{v}'_i \Delta t
$$

The previous steps solve the equations of motion using
Euler's scheme[^1].

[^1]: Euler's integration is numerically unstable on this problem, and
      therefore more accurate but more complex schemes are used in
      practice.

In this program we trade accuracy for simplicity by ignoring the
factor $G m_i m_j$ and rewriting the sum as:

$$
\textbf{F}_i := \sum_{j = 0}^{N-1} \frac{\textbf{d}_{ij}}{(d_{ij}^2 + \epsilon)^{3/2}}
$$

where $\textbf{d}_{ij}$ is the vector from particle $i$ to particle
$j$, i.e., $\textbf{d}_{ij} := (\textbf{x}_j - \textbf{x}_i)$, and
$\epsilon > 0$ is used to avoid a division by zero when $i = j$, that
is, when computing the interaction of a particle with itself.

***/

/***
Modify the provided program to use the GPU. A first version can be
easily obtained from the serial version.

Then, observe that the data of each particle is (re)read from device
memory $N$ times; indeed, each of the $N$ CUDA threads scans the
entire `p[]` array, so each element of `p[]` is accessed $N$ times
by $N$ different threads.

![Figure 1: Using _shared memory_](cuda-nbody.svg)

In situations like this, it can be useful to try to use use shared
memory to reduce access to the device memory (Figure 1). To do this:

a. Each thread block copies a block of elements from `p[]` in global
   memory in an array `tmp[]` in shared memory;

b. Each thread computes the total force acting a particle
   $p_i$. Hence, CUDA thread $i$ compute the interaction between $p_i$
   and all particles in `tmp[]`;

c. Once the previous step is finished, the program copies the next
   block of particles from device memory to the `tmp[]` array

d. Each CUDA thread computes the force exerted on particle $p_i$ by
   the new particles in `tmp[]`.

The previous process is repeated until all the particles in `p[]` have
been taken into account. In this way, each element of `p[]` is reread
multiple times (once for each block), instead of $N$ times. Since the
number of blocks is less than $N$, this reduces the pressure on the
memory of the device, improving the performance of the program.

To compile without shared memory:

        nvcc cuda-nbody.cu -o cuda-nbody -lm

To compile with shared memory:

        nvcc -DUSE_SHARED cuda-nbody.cu -o cuda-nbody -lm

To execute:

        ./cuda-nbody [nsteps [nparticles]]

Example:

        ./cuda-nbody 50 15000

## Files

- [cuda-nbody.cu](cuda-nbody.cu) [hpc.h](hpc.h)

***/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define BLKDIM 1024

const double EPSILON = 1.0e-9f;
/* const double G = 6.67e-11; */

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
void init(float3 *x, float3 *v, int n)
{
    for (int i = 0; i < n; i++) {
        x[i].x = randab(-1, 1);
        x[i].y = randab(-1, 1);
        x[i].z = randab(-1, 1);
        v[i].x = randab(-1, 1);
        v[i].y = randab(-1, 1);
        v[i].z = randab(-1, 1);
    }
}

/**
 * Compute the new velocities of all particles in `p`
 */
#ifdef SERIAL
void compute_force(const float3 *x, float3 *v, float dt, int n)
{
    for (int i = 0; i < n; i++) {
        float3 F = {0.0, 0.0, 0.0};

        for (int j = 0; j < n; j++) {
            const float dx = x[j].x - x[i].x;
            const float dy = x[j].y - x[i].y;
            const float dz = x[j].z - x[i].z;
            const float distSqr = dx*dx + dy*dy + dz*dz + EPSILON;
            const float invDist = 1.0f / sqrtf(distSqr);
            const float invDist3 = invDist * invDist * invDist;

            F.x += dx * invDist3;
            F.y += dy * invDist3;
            F.z += dz * invDist3;
        }
        v[i].x += dt*F.x;
        v[i].y += dt*F.y;
        v[i].z += dt*F.z;
    }
}
#else
#ifdef USE_SHARED
__device__ int d_min(int a, int b)
{
    return (a < b ? a : b);
}

__global__ void compute_force(const float3 *x, float3 *v, float dt, int n)
{
    /* This version uses shared memory */
    __shared__ float3 tmp[BLKDIM];
    const int li = threadIdx.x;
    const int gi = threadIdx.x + blockIdx.x * blockDim.x;
    float3 F = {0.0, 0.0, 0.0};

    for (int b = 0; b < n; b += BLKDIM) {

        /* Care should be taken if the number of particles is not
           a multiple of BLKDIM */
        const int DIM = d_min(n - b, BLKDIM);

        if (li < DIM)
            tmp[li] = x[b + li];

        /* Wait for all threads to fill the shared memory */
        __syncthreads();

        if (gi < n) {
            for (int j = 0; j < DIM; j++) {
                const float dx = tmp[j].x - x[gi].x;
                const float dy = tmp[j].y - x[gi].y;
                const float dz = tmp[j].z - x[gi].z;

                const float distSqr = dx*dx + dy*dy + dz*dz + EPSILON;
                const float invDist = 1.0f / sqrtf(distSqr);
                const float invDist3 = invDist * invDist * invDist;

                F.x += dx * invDist3;
                F.y += dy * invDist3;
                F.z += dz * invDist3;
            }
        }

        /* Wait for all threads to finish the computation before
           modifying the shared memory at the next iteration */
        __syncthreads();
    }
    if (gi < n) {
        v[gi].x += dt*F.x;
        v[gi].y += dt*F.y;
        v[gi].z += dt*F.z;
    }
}
#else
__global__ void compute_force(const float3 *x, float3 *v, float dt, int n)
{
    /* This version does not use shared memory */
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    float3 F = {0.0, 0.0, 0.0};

    if (i<n) {
        for (int j = 0; j < n; j++) {
            const float dx = x[j].x - x[i].x;
            const float dy = x[j].y - x[i].y;
            const float dz = x[j].z - x[i].z;
            const float distSqr = dx*dx + dy*dy + dz*dz + EPSILON;
            const float invDist = 1.0f / sqrtf(distSqr);
            const float invDist3 = invDist * invDist * invDist;

            F.x += dx * invDist3;
            F.y += dy * invDist3;
            F.z += dz * invDist3;
        }
        v[i].x += dt*F.x;
        v[i].y += dt*F.y;
        v[i].z += dt*F.z;
    }
}
#endif
#endif

/**
 * Update the positions of all particles in p using the updated
 * velocities.
 */
#ifdef SERIAL
void integrate_positions(float3 *x, const float3 *v, float dt, int n)
{
    for (int i = 0 ; i < n; i++) {
        x[i].x += v[i].x*dt;
        x[i].y += v[i].y*dt;
        x[i].z += v[i].z*dt;
    }
}
#else
__global__ void integrate_positions(float3 *x, const float3 *v, float dt, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i].x += v[i].x*dt;
        x[i].y += v[i].y*dt;
        x[i].z += v[i].z*dt;
    }
}
#endif

float energy(const float3 *x, const float3 *v, int n)
{
    float energy = 0.0;
    /* The kinetic energy of an n-body system is:

       K = (1/2) * sum_i [m_i * (vx_i^2 + vy_i^2 + vz_i^2)]

    */

    for (int i=0; i<n; i++) {
        energy += 0.5*(v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z);
        /* Accumulate potential energy, defined as

           sum_{i<j} - m[j] * m[j] / d_ij

         */
        for (int j=i+1; j<n; j++) {
            const float dx = x[i].x - x[j].x;
            const float dy = x[i].y - x[j].y;
            const float dz = x[i].z - x[j].z;
            const float distance = sqrt(dx*dx + dy*dy + dz*dz);
            energy -= 1.0f / distance;
        }
    }
    return energy;
}

int main(int argc, char* argv[])
{
    int nBodies = 10000;
    const int MAXBODIES = 50000;
    int nIters = 10;
    const float DT = 1e-6; /* time step */
    float3 *x, *v;
#ifndef SERIAL
    float3 *d_x, *d_v;
#endif

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [nbodies [niter]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        nBodies = atoi(argv[1]);
    }

    if (nBodies > MAXBODIES) {
        fprintf(stderr, "FATAL: too many bodies\n");
        return EXIT_FAILURE;
    }

    if (argc > 2) {
        nIters = atoi(argv[2]);
    }

    srand(1234);

    printf("%d particles, %d steps\n", nBodies, nIters);
#ifndef SERIAL
#ifdef USE_SHARED
    printf("using shared memory\n");
#else
    printf("NOT using shared memory\n");
#endif
#endif

    const size_t size = nBodies*sizeof(float3);
    x = (float3*)malloc(size); assert(x != NULL);
    v = (float3*)malloc(size); assert(v != NULL);

    init(x, v, nBodies); /* Init pos / vel data */

#ifndef SERIAL
    cudaSafeCall( cudaMalloc((void**)&d_x, size) );
    cudaSafeCall( cudaMalloc((void**)&d_v, size) );
    cudaSafeCall( cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice) );

    const dim3 BLOCK(BLKDIM);
    const dim3 GRID((nBodies + BLKDIM - 1)/BLKDIM);
#endif

    float totalTime = 0.0;
    for (int iter = 1; iter <= nIters; iter++) {
        const double tstart = hpc_gettime();
#ifdef SERIAL
        compute_force(x, v, DT, nBodies);
        integrate_positions(x, v, DT, nBodies);
#else
        compute_force<<<GRID, BLOCK>>>(d_x, d_v, DT, nBodies);
        cudaCheckError();
        integrate_positions<<<GRID, BLOCK>>>(d_x, d_v, DT, nBodies);
        cudaCheckError();

        cudaDeviceSynchronize();
#endif
        const double elapsed = hpc_gettime() - tstart;
        totalTime += elapsed;
#ifndef SERIAL
        /* The following copy is required to compute the energy on the
           CPU. It would be possible to compute the energy on the GPU,
           so that this copy operation would not be required */
        cudaSafeCall( cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost) );
        cudaSafeCall( cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost) );
#endif
        printf("Iteration %3d/%3d : energy=%f, %.3f seconds\n", iter, nIters, energy(x, v, nBodies), elapsed);
        fflush(stdout);
    }
    const double avgTime = totalTime / nIters;

    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

    free(x);
    free(v);
#ifndef SERIAL
    cudaFree(d_x);
    cudaFree(d_v);
#endif
    return EXIT_SUCCESS;
}
