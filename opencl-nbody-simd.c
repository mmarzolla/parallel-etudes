/****************************************************************************
 *
 * opencl-nbody-simd.c
 *
 * N-body simulation with OpenCL (from https://github.com/harrism/mini-nbody)
 *
 * Copyright (C) Mark Harris
 * Last modified in 2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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

/***
% HPC - Simulazione N-body
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2021-12-02

Implementazione della simulazione N-body usando tipi di dato SIMD sul
device.

> **NOTA** per qualche motivo che non ho compreso, produce un
> risultato diverso dalla versione non SIMD. Indagare.

## File

- [opencl-nbody-simd.c](opencl-nbody-simd.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "simpleCL.h"

const float EPSILON = 1.0e-9f;
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
void init(cl_float3 *x, cl_float3 *v, int n)
{
    for (int i = 0; i < n; i++) {
        x[i].s[0] = randab(-1,1);
        x[i].s[1] = randab(-1,1);
        x[i].s[2] = randab(-1,1);
        v[i].s[0] = randab(-1,1);
        v[i].s[1] = randab(-1,1);
        v[i].s[2] = randab(-1,1);
    }
}

#ifdef SERIAL
/**
 * Compute the new velocities of all particles in `p`
 */
void compute_force(const cl_float3 *x, cl_float3 *v, float dt, int n)
{
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            const float dx = x[j].s[0] - x[i].s[0];
            const float dy = x[j].s[1] - x[i].s[1];
            const float dz = x[j].s[2] - x[i].s[2];
            const float distSqr = dx*dx + dy*dy + dz*dz + EPSILON;
            const float invDist = 1.0f / sqrtf(distSqr);
            const float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        v[i].s[0] += dt*Fx;
        v[i].s[1] += dt*Fy;
        v[i].s[2] += dt*Fz;
    }
}

/**
 * Update the positions of all particles in p using the updated
 * velocities.
 */
void integrate_positions(cl_float3 *x, const cl_float3 *v, float dt, int n)
{
    for (int i = 0 ; i < n; i++) {
        x[i].s[0] += v[i].s[0]*dt;
        x[i].s[1] += v[i].s[1]*dt;
        x[i].s[2] += v[i].s[2]*dt;
    }
}

float energy(const cl_float3 *x, const cl_float3 *v, int n)
{
    float energy = 0.0;
    /* The kinetic energy of an n-body system is:

       K = (1/2) * sum_i [m_i * (vx_i^2 + vy_i^2 + vz_i^2)]

    */

    for (int i=0; i < n; i++) {
        energy += (v[i].s[0] * v[i].s[0]) +
                  (v[i].s[1] * v[i].s[1]) +
                  (v[i].s[2] * v[i].s[2]);
        /* Accumulate potential energy, defined as

           sum_{i<j} - m[j] * m[j] / d_ij

         */
        for (int j=i+1; j<n; j++) {
            const float dx = x[i].s[0] - x[j].s[0];
            const float dy = x[i].s[1] - x[j].s[1];
            const float dz = x[i].s[2] - x[j].s[2];
            const float distance = sqrt(dx*dx + dy*dy + dz*dz);
            energy -= 1.0f / distance;
        }
    }
    return energy;
}
#endif

int main(int argc, char* argv[])
{
    int nBodies = 10000;
    const int MAXBODIES = 50000;
    int nIters = 10;
    const float DT = 1e-6f; /* time step */
    cl_float3 *x, *v;
    float energy;
#ifndef SERIAL
    cl_mem d_x, d_v, d_energies;
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

    const size_t size = nBodies*sizeof(cl_float3);
    x = (cl_float3*)malloc(size); assert(x != NULL);
    v = (cl_float3*)malloc(size); assert(v != NULL);

    init(x, v, nBodies); /* Init pos / vel data */

#ifndef SERIAL
    sclInitFromFile("opencl-nbody-simd.cl");
    sclDim grid, block;
    sclWGSetup1D(nBodies, &grid, &block);
    const size_t N_OF_BLOCKS = grid.sizes[0] / block.sizes[0];

    d_x = sclMallocCopy(size, x, CL_MEM_READ_WRITE);
    d_v = sclMallocCopy(size, v, CL_MEM_READ_WRITE);

    float energies[N_OF_BLOCKS];
    d_energies = sclMalloc(sizeof(energies), CL_MEM_READ_WRITE);
    sclKernel compute_force_kernel = sclCreateKernel("compute_force_kernel");
    sclKernel integrate_positions_kernel = sclCreateKernel("integrate_positions_kernel");
    sclKernel energy_kernel = sclCreateKernel("energy_kernel");
#endif

    float totalTime = 0.0;
    for (int iter = 1; iter <= nIters; iter++) {
        const double tstart = hpc_gettime();
#ifdef SERIAL
        compute_force(x, v, DT, nBodies);
        integrate_positions(x, v, DT, nBodies);
#else
        sclSetArgsEnqueueKernel(compute_force_kernel,
                                grid, block,
                                ":b :b :f :d",
                                d_x, d_v, DT, nBodies);
        sclSetArgsEnqueueKernel(integrate_positions_kernel,
                                grid, block,
                                ":b :b :f :d",
                                d_x, d_v, DT, nBodies);
        sclDeviceSynchronize();
#endif
        const double elapsed = hpc_gettime() - tstart;
        totalTime += elapsed;
#ifndef SERIAL
        /* The following copy is required to compute the energy on the
           CPU. It would be possible to compute the energy on the GPU,
           so that this copy operation would not be required */

        sclSetArgsEnqueueKernel(energy_kernel,
                                grid, block,
                                ":b :b :d :b",
                                d_x, d_v, nBodies, d_energies);
        sclMemcpyDeviceToHost(energies, d_energies, sizeof(energies));
        energy = 0.0f;
        for (int i=0; i<N_OF_BLOCKS; i++) {
            energy += energies[i];
        }
#else
        energy = energy(v, nBodies);
#endif
        printf("Iteration %3d/%3d : energy=%f, %.3f seconds\n", iter, nIters, energy, elapsed);
        fflush(stdout);
    }
    const double avgTime = totalTime / nIters;

    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

    free(x);
    free(v);
#ifndef SERIAL
    sclFree(d_x);
    sclFree(d_v);
    sclFree(d_energies);
#endif
    return EXIT_SUCCESS;
}
