/****************************************************************************
 *
 * mpi-nbody.c -- N-body simulation (from https://github.com/harrism/mini-nbody)
 *
 * Copyright (C) Mark Harris
 * Modified 2021, 2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - N-body simulation
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-09-06

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
Modify the serial program to make use of OpenMP parallelism.

To compile:

		mpicc -std=c99 -Wall -Wpedantic mpi-nbody.c -o mpi-nbody -lm

To execute:

        mpirun -n 4 ./mpi-nbody [nsteps [nparticles]]

Example:

        mpirun -n 4 ./mpi-nbody 50 10000

## File

- [mpi-nbody.c](mpi-nbody.c)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for sqrtf() */
#include <assert.h>
#include <mpi.h>

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

/**
 * Compute the total energy of the system as the sum of kinetic and
 * potential energy.
 */
float energy(const float *x, const float *y, const float *z,
             const float *vx, const float *vy, const float *vz, int n)
{
    float e = 0.0;
    /* The kinetic energy of an n-body system is:

       K = (1/2) * sum_i [m_i * (vx_i^2 + vy_i^2 + vz_i^2)]

    */
    for (int i=0; i<n; i++) {
        e += 0.5 * (vx[i] * vx[i]) + (vy[i] * vy[i]) + (vz[i] * vz[i]);
        /* Accumulate potential energy, defined as

           sum_{i<j} - m[j] * m[j] / d_ij

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
	int my_rank, comm_sz;
    int nBodies = 10000;
    const int MAXBODIES = 50000;
    int nIters = 10;
    const float DT = 1e-6f; /* time step */
    float *x, *y, *z, *vx, *vy, *vz;
    float *my_x, *my_y, *my_z, *my_vx, *my_vy, *my_vz;
    
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

    const size_t size = nBodies*sizeof(float);
    const size_t my_size = size / comm_sz;
    const long my_nBodies = nBodies / comm_sz;
    x = (float*)malloc(size); assert(x != NULL);
    y = (float*)malloc(size); assert(y != NULL);
    z = (float*)malloc(size); assert(z != NULL);
    vx = (float*)malloc(size); assert(vx != NULL);
    vy = (float*)malloc(size); assert(vy != NULL);
    vz = (float*)malloc(size); assert(vz != NULL);
    
    my_x = (float*)malloc(my_size); assert(my_x != NULL);
    my_y = (float*)malloc(my_size); assert(my_y != NULL);
    my_z = (float*)malloc(my_size); assert(my_z != NULL);
    my_vx = (float*)malloc(my_size); assert(my_vx != NULL);
    my_vy = (float*)malloc(my_size); assert(my_vy != NULL);
    my_vz = (float*)malloc(my_size); assert(my_vz != NULL);

	if ( 0 == my_rank ) {
		init(x, y, z, vx, vy, vz, nBodies); /* Init pos / vel data */
	}
	
	float totalTime = 0.0;
	
	MPI_Scatter( x,				/* sendbuf      	 	 */
				 my_nBodies,	/* count; how many elements to send to _each_ destination */
				 MPI_FLOAT,		/* sent datatype 	 	 */
				 my_x,			/* recvbuf      	 	 */
				 my_nBodies,	/* recvcount    	 	 */
				 MPI_FLOAT,		/* received datatype 	 */
				 0,				/* source       	 	 */
				 MPI_COMM_WORLD /* communicator 	 	 */
				 );
	MPI_Scatter( y,				/* sendbuf      	 	 */
				 my_nBodies,	/* count; how many elements to send to _each_ destination */
				 MPI_FLOAT,		/* sent datatype 	 	 */
				 my_y,			/* recvbuf      	 	 */
				 my_nBodies,	/* recvcount    	 	 */
				 MPI_FLOAT,		/* received datatype 	 */
				 0,				/* source       	 	 */
				 MPI_COMM_WORLD /* communicator 	 	 */
				 );
	MPI_Scatter( z,				/* sendbuf      	 	 */
				 my_nBodies,	/* count; how many elements to send to _each_ destination */
				 MPI_FLOAT,		/* sent datatype 	 	 */
				 my_z,			/* recvbuf      	 	 */
				 my_nBodies,	/* recvcount    	 	 */
				 MPI_FLOAT,		/* received datatype 	 */
				 0,				/* source       	 	 */
				 MPI_COMM_WORLD /* communicator 	 	 */
				 );
				 
	MPI_Scatter( vx,			/* sendbuf      	 	 */
				 my_nBodies,	/* count; how many elements to send to _each_ destination */
				 MPI_FLOAT,		/* sent datatype 	 	 */
				 my_vx,			/* recvbuf      	 	 */
				 my_nBodies,	/* recvcount    	 	 */
				 MPI_FLOAT,		/* received datatype 	 */
				 0,				/* source       	 	 */
				 MPI_COMM_WORLD /* communicator 	 	 */
				 );
	MPI_Scatter( vy,			/* sendbuf      	 	 */
				 my_nBodies,	/* count; how many elements to send to _each_ destination */
				 MPI_FLOAT,		/* sent datatype 	 	 */
				 my_vy,			/* recvbuf      	 	 */
				 my_nBodies,	/* recvcount    	 	 */
				 MPI_FLOAT,		/* received datatype 	 */
				 0,				/* source       	 	 */
				 MPI_COMM_WORLD /* communicator 	 	 */
				 );
	MPI_Scatter( vz,			/* sendbuf      	 	 */
				 my_nBodies,	/* count; how many elements to send to _each_ destination */
				 MPI_FLOAT,		/* sent datatype 	 	 */
				 my_vz,			/* recvbuf      	 	 */
				 my_nBodies,	/* recvcount    	 	 */
				 MPI_FLOAT,		/* received datatype 	 */
				 0,				/* source       	 	 */
				 MPI_COMM_WORLD /* communicator 	 	 */
				 );
	
    for (int iter = 1; iter <= nIters; iter++) {

        const double tstart = hpc_gettime();
        compute_force(my_x, my_y, my_z, my_vx, my_vy, my_vz, DT, my_nBodies);
        integrate_positions(my_x, my_y, my_z, my_vx, my_vy, my_vz, DT, my_nBodies);
        const double elapsed = hpc_gettime() - tstart;
        totalTime += elapsed;
        
        MPI_Gather( my_x,			/* sendbuf      	 	 */
					my_nBodies,		/* count; how many elements to send to _each_ destination */
					MPI_FLOAT,		/* sent datatype 	 	 */
					x,				/* recvbuf      	 	 */
					my_nBodies,		/* recvcount    	 	 */
					MPI_FLOAT,		/* received datatype 	 */
					0,				/* source       	 	 */
					MPI_COMM_WORLD	/* communicator 	 	 */
					);
		MPI_Gather( my_y,			/* sendbuf      	 	 */
					my_nBodies,		/* count; how many elements to send to _each_ destination */
					MPI_FLOAT,		/* sent datatype 	 	 */
					y,				/* recvbuf      	 	 */
					my_nBodies,		/* recvcount    	 	 */
					MPI_FLOAT,		/* received datatype 	 */
					0,				/* source       	 	 */
					MPI_COMM_WORLD	/* communicator 	 	 */
					);
		MPI_Gather( my_z,			/* sendbuf      	 	 */
					my_nBodies,		/* count; how many elements to send to _each_ destination */
					MPI_FLOAT,		/* sent datatype 	 	 */
					z,				/* recvbuf      	 	 */
					my_nBodies,		/* recvcount    	 	 */
					MPI_FLOAT,		/* received datatype 	 */
					0,				/* source       	 	 */
					MPI_COMM_WORLD	/* communicator 	 	 */
					);
		
		MPI_Gather( my_vx,			/* sendbuf      	 	 */
					my_nBodies,		/* count; how many elements to send to _each_ destination */
					MPI_FLOAT,		/* sent datatype 	 	 */
					vx,				/* recvbuf      	 	 */
					my_nBodies,		/* recvcount    	 	 */
					MPI_FLOAT,		/* received datatype 	 */
					0,				/* source       	 	 */
					MPI_COMM_WORLD	/* communicator 	 	 */
					);
		MPI_Gather( my_vy,			/* sendbuf      	 	 */
					my_nBodies,		/* count; how many elements to send to _each_ destination */
					MPI_FLOAT,		/* sent datatype 	 	 */
					vy,				/* recvbuf      	 	 */
					my_nBodies,		/* recvcount    	 	 */
					MPI_FLOAT,		/* received datatype 	 */
					0,				/* source       	 	 */
					MPI_COMM_WORLD	/* communicator 	 	 */
					);
		MPI_Gather( my_vz,			/* sendbuf      	 	 */
					my_nBodies,		/* count; how many elements to send to _each_ destination */
					MPI_FLOAT,		/* sent datatype 	 	 */
					vz,				/* recvbuf      	 	 */
					my_nBodies,		/* recvcount    	 	 */
					MPI_FLOAT,		/* received datatype 	 */
					0,				/* source       	 	 */
					MPI_COMM_WORLD	/* communicator 	 	 */
					);
        
        if ( 0 == my_rank ) {

			const float e = energy(x, y, z, vx, vy, vz, nBodies);
			printf("Iteration %3d/%3d : E=%f, %.3f seconds\n", iter, nIters, e, elapsed);
			fflush(stdout);
		}
    }
    
    if ( 0 == my_rank ) {		
		const double avgTime = totalTime / nIters;

		printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
	}
	
    free(x); free(y); free(z);
    free(my_x); free(my_y); free(my_z);
    free(vx); free(vy); free(vz);
    free(my_vx); free(my_vy); free(my_vz);
    
    MPI_Finalize();
    
    return EXIT_SUCCESS;
}