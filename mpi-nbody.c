/****************************************************************************
 *
 * mpi-nbody.c - N-body simulation
 *
 * Copyright (C) Mark Harris
 * Copyright (C) 2021--2024 Moreno Marzolla
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
% N-body simulation
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-11-08

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
Modify the serial program to make use of distributed-memory parallelism.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-nbody.c -o mpi-nbody -lm

To execute:

        mpirun -n [P] ./mpi-nbody [nsteps [nparticles]]

## File

- [mpi-nbody.c](mpi-nbody.c) [hpc.h](hpc.h)

***/
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for sqrtf() */
#include <assert.h>
#include <mpi.h>

int my_rank, comm_sz;

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
 * stored in `p`. This function must be called by the rank 0 process.
 */
void init(float *x, float *y, float *z,
          float *vx, float *vy, float *vz,
          int n)
{
    assert( 0 == my_rank );
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
 * Compute the new velocities of the particles that are handled by the
 * current process. Requires updated positions of all particles.
 */
void update_velocities(const float *x, const float *y, const float *z,
                       float *vx, float *vy, float *vz,
                       float dt, int n)
{
    const int start = (n * my_rank) / comm_sz;
    const int end = (n * (my_rank+1)) / comm_sz;

    for (int i = start; i < end; i++) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            const float dx = x[j] - x[i];
            const float dy = y[j] - y[i];
            const float dz = z[j] - z[i];
            const float distSqr = dx*dx + dy*dy + dz*dz + EPSILON;
            const float invDist = 1.0f / sqrtf(distSqr);
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
 * Update the positions of the particles that are handled by the
 * current process, using the updated velocities.
 */
void update_positions(float *x, float *y, float *z,
                      const float *vx, const float *vy, const float *vz,
                      float dt, int n)
{
    const int start = (n * my_rank) / comm_sz;
    const int end = (n * (my_rank+1)) / comm_sz;

    for (int i = start ; i < end; i++) {
        x[i] += vx[i]*dt;
        y[i] += vy[i]*dt;
        z[i] += vz[i]*dt;
    }
}

/**
 * Compute the energy of the particles that are handled by the current
 * process. Requires the updated positions of ALL particles, and
 * updated velocities of the particles handled by the current process
 * only.
 */
float energy(const float *x, const float *y, const float *z,
             const float *vx, const float *vy, const float *vz,
             int n)
{
    const int start = (n * my_rank) / comm_sz;
    const int end = (n * (my_rank+1)) / comm_sz;

    float e = 0.0;
    /* The kinetic energy of an n-body system is:

       K = (1/2) * sum_i [m_i * (vx_i^2 + vy_i^2 + vz_i^2)]

    */
    for (int i=start; i<end; i++) {
        e += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
        /* Accumulate potential energy, defined as

           sum_{i<j} - m[i] * m[j] / d_ij

         */
        for (int j=i+1; j<n; j++) {
            const float dx = x[i] - x[j];
            const float dy = y[i] - y[j];
            const float dz = z[i] - z[j];
            const float distance = sqrt(dx*dx + dy*dy + dz*dz);
            e -= 1.0f / distance;
        }
    }
    return e;
}

int main(int argc, char* argv[])
{
    int nBodies = 10000;
    const int MAXBODIES = 50000;
    int nIters = 10;
    const float DT = 1e-6f; /* time step */
    float *x, *y, *z, *vx, *vy, *vz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

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

    if ( 0 == my_rank ) {
        printf("%d particles, %d steps\n", nBodies, nIters);
    }

    int sendcounts[comm_sz];
    int displs[comm_sz];
    for (int i=0; i<comm_sz; i++) {
        const int rank_start = (nBodies * i) / comm_sz;
        const int rank_end = nBodies * (i + 1) / comm_sz;
        sendcounts[i] = rank_end - rank_start;
        displs[i] = rank_start;
    }

    const int my_nBodies = sendcounts[my_rank];
    const size_t size = nBodies*sizeof(float);

    x = (float*)malloc(size); assert(x != NULL);
    y = (float*)malloc(size); assert(y != NULL);
    z = (float*)malloc(size); assert(z != NULL);
    vx = (float*)malloc(size); assert(vx != NULL);
    vy = (float*)malloc(size); assert(vy != NULL);
    vz = (float*)malloc(size); assert(vz != NULL);

    if ( 0 == my_rank ) {
        init(x, y, z, vx, vy, vz, nBodies); /* Init pos / vel data */
    }

    /* NOTE: in-place operations */
    MPI_Scatterv( vx,            /* sendbuf             */
                  sendcounts,    /* sendcounts          */
                  displs,        /* displacements       */
                  MPI_FLOAT,     /* sent datatype       */
                  (my_rank == 0 ? MPI_IN_PLACE : vx), /* recvbuf */
                  my_nBodies,    /* recvcount           */
                  MPI_FLOAT,     /* received datatype   */
                  0,             /* root                */
                  MPI_COMM_WORLD /* communicator        */
                  );

    MPI_Scatterv( vy,            /* sendbuf             */
                  sendcounts,    /* sendcounts          */
                  displs,        /* displacements       */
                  MPI_FLOAT,     /* sent datatype       */
                  (my_rank == 0 ? MPI_IN_PLACE : vy), /* recvbuf */
                  my_nBodies,    /* recvcount           */
                  MPI_FLOAT,     /* received datatype   */
                  0,             /* root                */
                  MPI_COMM_WORLD /* communicator        */
                  );
    MPI_Scatterv( vz,            /* sendbuf             */
                  sendcounts,    /* sendcounts          */
                  displs,        /* displacements       */
                  MPI_FLOAT,     /* sent datatype       */
                  (my_rank == 0 ? MPI_IN_PLACE : vz), /* recvbuf */
                  my_nBodies,    /* recvcount           */
                  MPI_FLOAT,     /* received datatype   */
                  0,             /* root                */
                  MPI_COMM_WORLD /* communicator        */
                  );

    MPI_Bcast( x,                /* sendbuf             */
               nBodies,          /* count               */
               MPI_FLOAT,        /* sent datatype       */
               0,                /* root                */
               MPI_COMM_WORLD    /* communicator        */
               );
    MPI_Bcast( y,                /* sendbuf             */
               nBodies,          /* count               */
               MPI_FLOAT,        /* sent datatype       */
               0,                /* root                */
               MPI_COMM_WORLD    /* communicator        */
               );
    MPI_Bcast( z,                /* sendbuf             */
               nBodies,          /* count               */
               MPI_FLOAT,        /* sent datatype       */
               0,                /* root                */
               MPI_COMM_WORLD    /* communicator        */
               );

    const double tstart_global = MPI_Wtime();

    for (int iter = 1; iter <= nIters; iter++) {
        const double tstart = MPI_Wtime();
        update_velocities(x, y, z, vx, vy, vz, DT, nBodies);
        update_positions(x, y, z, vx, vy, vz, DT, nBodies);

        /* Note: in-place operations */
        MPI_Allgatherv( MPI_IN_PLACE,   /* sendbuf              */
                        0,              /* ignored              */
                        MPI_DATATYPE_NULL, /* ignored           */
                        x,              /* recvbuf              */
                        sendcounts,     /* recvcount            */
                        displs,         /* displacements        */
                        MPI_FLOAT,      /* received type        */
                        MPI_COMM_WORLD  /* communicator         */
                        );
        MPI_Allgatherv( MPI_IN_PLACE,   /* sendbuf              */
                        0,              /* ignored              */
                        MPI_DATATYPE_NULL, /* ignored           */
                        y,              /* recvbuf              */
                        sendcounts,     /* recvcount            */
                        displs,         /* displacements        */
                        MPI_FLOAT,      /* received datatype    */
                        MPI_COMM_WORLD  /* communicator         */
                        );
        MPI_Allgatherv( MPI_IN_PLACE,   /* sendbuf              */
                        0,              /* ignored              */
                        MPI_DATATYPE_NULL, /* ignored           */
                        z,              /* recvbuf              */
                        sendcounts,     /* recvcount            */
                        displs,         /* displacements        */
                        MPI_FLOAT,      /* received datatype    */
                        MPI_COMM_WORLD  /* communicator         */
                        );

        const float local_e = energy(x, y, z, vx, vy, vz, nBodies);
        float e = 0.0f;

        MPI_Reduce( &local_e,           /* send buffer          */
                    &e,                 /* receive buffer       */
                    1,                  /* count                */
                    MPI_FLOAT,          /* datatype             */
                    MPI_SUM,            /* operation            */
                    0,                  /* destination          */
                    MPI_COMM_WORLD      /* communicator         */
                    );

        if ( 0 == my_rank ) {
            const double elapsed = MPI_Wtime() - tstart;

            printf("Iteration %3d/%3d : E=%f, %.3f seconds\n", iter, nIters, e, elapsed);
            fflush(stdout);
        }
    }

    if ( 0 == my_rank ) {
        const double totalTime = MPI_Wtime() - tstart_global;
        const double avgTime = totalTime / nIters;
        printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
    }

    free(x); free(y); free(z);
    free(vx); free(vy); free(vz);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
