/****************************************************************************
 *
 * omp-nbody-task.c - N-body simulation with tasks
 *
 * Copyright (C) 2026 Moreno Marzolla
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
% N-body simulation with tasks
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-04-28

![A frame of the Bolshoi simulation (source: <http://hipacc.ucsc.edu/Bolshoi/Images.html>).](bolshoi.png)

Cosmological simulations study the large-scale evolution of the
universe, and are based on the computation of the dynamics of $N$
masses subject to mutual gravitational attraction ($N$-body
problem). The [Bolshoi
simulation](http://hipacc.ucsc.edu/Bolshoi.html) required 6 million
CPU hours on one of the most powerful supercomputers of 2010. In this
exercise we solve a $N$-body problem for a small $N$ using a simple
algorithm, based on a program developed by Mark Harris available at
<https://github.com/harrism/mini-nbody>.

The physical laws governing the dynamics of $N$ masses were discovered
by [Isaac Newton](https://en.wikipedia.org/wiki/Isaac_Newton): these
are the [second law of
dynamic](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion#Newton%27s_second_law)
and the [law of universal
gravitation](https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation).

The second law of dynamics states that a force $\textbf{F}$ acting on
a point mass $m$ produces an acceleration $\textbf{a}$ such that
$\textbf{F}=m\textbf{a}$. The law of universal gravitation states that
two point masses $m_i$ and $m_j$ at a distance $d_{ij}$ are subject to
an attractive force of magnitude $F_{ij} = G m_i m_j / d_{ij}^2$ where
$G$ is the gravitational constant ($G \approx 6,674 \times 10^{-11}\
\mathrm{N}\ \mathrm{m}^2\ \mathrm{kg}^{-2}$).

The following explanation is not required for solving this exercise,
but might be informative, and only requires basic physics knowledge.

Let us consider $N$ point masses $m_0, \ldots, m_{n-1}$ that are
subject to mutual gravitational attraction. Since the masses are
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
squared distance between particles $i$ and $j$, and
$\textbf{n}_{ij}$ is the unit vector from particle $i$ to $j$

2. Compute the acceleration $\textbf{a}_i$ acting on mass $i$:
$$
\textbf{a}_i := \textbf{F}_i / m_i
$$

3. Compute the new velocity $\textbf{v}'_i$ of mass $i$ at time $t'$:
$$
\textbf{v}'_i := \textbf{v}_i + \textbf{a}_i \Delta t
$$

4. Compute the new position $\textbf{x}'_i$ of mass $i$ at time $t'$:
$$
\textbf{x}'_i := \textbf{x}_i + \textbf{v}'_i \Delta t
$$

The previous steps solve the equations of motion using
Euler's scheme[^1].

[^1]: Euler's integration is not recommended on this kind or problem,
      since it might be numerically unstable; more accurate and
      complex schemes are used in practice.

In this program we trade accuracy for simplicity by ignoring the
factor $G m_i m_j$ and rewriting the sum as:

$$
\textbf{F}_i := \sum_{j = 0}^{N-1} \frac{\textbf{d}_{ij}}{(d_{ij}^2 + \epsilon)^{3/2}}
$$

where $\textbf{d}_{ij}$ is the vector from particle $i$ to particle
$j$, i.e., $\textbf{d}_{ij} := (\textbf{x}_j - \textbf{x}_i)$, and
$\epsilon > 0$ is used to avoid a division by zero when $i = j$.

# Task-parallel decomposition of N-body interactions

The computation of N-body interactions requires the computation of the
forces $\mathbf{F}_{ij}$ exerted on particle $i$ by particle $j$. Then
the total force exerted on $i$ is simply the sum $\mathbf{F}_i =
\sum_j \mathbf{F}_{ij}$.

Let us turn our attention on the matrix of forces:

$$
\left(
\begin{array}{cccc}
\mathbf{F}_{00} & \mathbf{F}_{01} & \ldots & \mathbf{F}_{0\ n-1}\\
\mathbf{F}_{10} & \mathbf{F}_{11} & \ldots & \mathbf{F}_{1\ n-1}\\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{F}_{n-1\ 0} & \mathbf{F}_{n-1\ 1} & \ldots & \mathbf{F}_{n-1\ n-1}
\end{array}
\right)
$$

Since gravitational interactions are symmetric, we need to compute
only half the forces. Storing the whole matrix requires space
$\Theta(n^2)$, but since we only need the row sums, we can reduce the
space to $\Theta(n)$. However, here lies the problem.

```C
for (int i=0; i<n; i++)
  F[i] = 0;

for (int i=0; i<n; i++) {
  for (int j=i+1; j<n; j++) {
    const float Fij = force(i, j);
    F[i] += Fij;
    F[j] += Fij;
  }
}
```

![Figure 1: Task decomposition of N-body simulation.](nbody-task.svg)

In this exercise we use a task-parallel decomposition of the N-body
simulation as shown in Figure 1.

First of all, we observe that gravitational interactions are
symmetric, so we only need to compute the upper triangular part of the
matrix (Figure 1.a). To avoid race conditions, we decompose a
triangular interaction into two triangles $T_1, T_2$ and a rectangle $S$
(Figure 1.b); $T_1$ and $T_2$ can be computed in parallel, and when they
are completed we can compute $S$. Finally, to compute a rectangle we
decompose it into four rectangles $S_1, S_2, S_3, S_4$; those of equal
color can be processed in parallel, so we can process $S_1, S_3$ in
parallel, and when they are done we process $S_2,S_4$ again in
parallel.

This can be expressed using two mutually recursive functions
`triangle()` and `rectangle()` below.

```C
void triangle(int imin, int jmin, int imax, int jmax);

void rectangle(int imin, int jmin, int imax, int jmax)
{
  if (imax - imin <= THRESHOLD)
    compute_all_pair_interactions();
  else {
    const int imid = (imax - imin)/2;
    const int jmid = (jmax - jmin)/2;
#pragma omp task
    rectangle(imin, jmin, imid, jmid);
#pragma omp task
    rectangle(imid, jmid, imax, jmax);
#pragma omp taskwait
#pragma omp task
    rectangle(imin, jmid, imid, jmax);
#pragma omp task
    rectangle(imid, jmin, imax, jmid);
#pragma omp taskwait
  }
}

void triangle(int imin, int jmin, int imax, int jmax)
{
  if (imax - imin <= THRESHOLD)
    compute_all_pair_interactions();
  else {
    const int imid = (imin + imax)/2;
    const int jmid = (jmin + jmax)/2;
#pragma omp task
    triangle(imin, jmin, imid, jmid);
#pragma omp task
    triangle(imid, jmid, imax, jmax);
#pragma omp taskwait
#pragma omp task
    rectangle(imin, jmid, imid, jmax);
#pragma omp taskwait
  }
}

```

Modify the serial program to make use of OpenMP parallelism.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-nbody-task.c -o omp-nbody-task -lm

To execute:

        ./omp-nbody-task [nparticles [nsteps]]

Example:

        ./omp-nbody-task 10000 50

## File

- [omp-nbody-task.c](omp-nbody-task.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for sqrtf() */
#include <assert.h>
#include <omp.h>

const int THRESHOLD = 32;
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
 * Compute the pairwise interaction is particles i and j; update
 * velocities of i and j accordingly.
 */
void interact(const float *x, const float *y, const float *z,
              float *vx, float *vy, float *vz,
              float dt,
              int i, int j)
{
    const float dx = x[j] - x[i];
    const float dy = y[j] - y[i];
    const float dz = z[j] - z[i];

    const float distSqr = dx*dx + dy*dy + dz*dz + EPSILON;
    const float invDist = 1.0f / sqrtf(distSqr);
    const float invDist3 = invDist * invDist * invDist;

    const float Fx = dx * invDist3;
    const float Fy = dy * invDist3;
    const float Fz = dz * invDist3;

    vx[i] += dt*Fx;
    vy[i] += dt*Fy;
    vz[i] += dt*Fz;

    vx[j] -= dt*Fx;
    vy[j] -= dt*Fy;
    vz[j] -= dt*Fz;
}

/* forward declaration */
void rectangle(const float *x, const float *y, const float *z,
               float *vx, float *vy, float *vz,
               float dt,
               int imin, int jmin, int imax, int jmax);

void triangle(const float *x, const float *y, const float *z,
              float *vx, float *vy, float *vz,
              float dt,
              int imin, int jmin, int imax, int jmax)
{
    assert(imin == jmin);
    assert(imax == jmax);
    if (imax - imin <= THRESHOLD) {
        for (int i = imin; i < imax; i++)
            for (int j = i+1; j < jmax; j++)
                interact(x, y, z, vx, vy, vz, dt, i, j);
    } else {
        const int imid = (imin + imax)/2;
        const int jmid = (jmin + jmax)/2;
#pragma omp task
        triangle(x, y, z, vx, vy, vz, dt, imin, jmin, imid, jmid);
#pragma omp task
        triangle(x, y, z, vx, vy, vz, dt, imid, jmid, imax, jmax);
#pragma omp taskwait
#pragma omp task
        rectangle(x, y, z, vx, vy, vz, dt, imin, jmid, imid, jmax);
#pragma omp taskwait
    }
}

void rectangle(const float *x, const float *y, const float *z,
               float *vx, float *vy, float *vz,
               float dt,
               int imin, int jmin, int imax, int jmax)
{
    if (imax - imin <= THRESHOLD) {
        for (int i = imin; i < imax; i++)
            for (int j = jmin; j<jmax; j++)
                interact(x, y, z, vx, vy, vz, dt, i, j);
    } else {
        const int imid = (imin + imax)/2;
        const int jmid = (jmin + jmax)/2;
#pragma omp task
        rectangle(x, y, z, vx, vy, vz, dt, imin, jmin, imid, jmid);
#pragma omp task
        rectangle(x, y, z, vx, vy, vz, dt, imid, jmid, imax, jmax);
#pragma omp taskwait
#pragma omp task
        rectangle(x, y, z, vx, vy, vz, dt, imin, jmid, imid, jmax);
#pragma omp task
        rectangle(x, y, z, vx, vy, vz, dt, imid, jmin, imax, jmid);
#pragma omp taskwait
    }
}

/**
 * Compute the new velocities of all particles in `p`
 */
void compute_force(const float *x, const float *y, const float *z,
                   float *vx, float *vy, float *vz,
                   float dt,
                   int n)
{
#pragma omp parallel
#pragma omp single
    triangle(x, y, z, vx, vy, vz, dt, 0, 0, n, n);
}

/**
 * Update the positions of all particles in p using the updated
 * velocities.
 */
void integrate_positions(float *x, float *y, float *z,
                         const float *vx, const float *vy, const float *vz,
                         float dt,
                         int n)
{
#ifndef SERIAL
#pragma omp parallel for default(none) shared(x, y, z, vx, vy, vz, n, dt)
#endif
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
             const float *vx, const float *vy, const float *vz,
             int n)
{
    double result = 0.0;
#ifndef SERIAL
#pragma omp parallel for default(none) shared(x, y, z, vx, vy, vz, n) reduction(+:result)
#endif
    for (int i=0; i<n; i++) {
        /* The kinetic energy of an n-body system is:

           K = (1/2) * sum_i [m_i * (vx_i^2 + vy_i^2 + vz_i^2)]

        */

        result += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
        /* Accumulate potential energy, defined as

           sum_{i<j} - m[j] * m[j] / d_ij

         */
        for (int j=i+1; j<n; j++) {
            const double dx = x[i] - x[j];
            const double dy = y[i] - y[j];
            const double dz = z[i] - z[j];
            const double distance = sqrtf(dx*dx + dy*dy + dz*dz);
            result -= 1.0 / distance;
        }
    }
    return result;
}

int main(int argc, char* argv[])
{
    int N = 10000;
    const int MAX_BODIES = 50000;
    int nsteps = 10;
    const float DT = 1e-6f; /* time step */
    float *x, *y, *z, *vx, *vy, *vz;

    if (argc > 3) {
        fprintf(stderr, "Usage: %s [nparticles [nsteps]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N > MAX_BODIES) {
        fprintf(stderr, "FATAL: too many bodies\n");
        return EXIT_FAILURE;
    }

    if (argc > 2) {
        nsteps = atoi(argv[2]);
    }

    srand(1234);

    printf("%d particles, %d steps\n", N, nsteps);

    const size_t size = N*sizeof(*x);
    x = (float*)malloc(size); assert(x != NULL);
    y = (float*)malloc(size); assert(y != NULL);
    z = (float*)malloc(size); assert(z != NULL);
    vx = (float*)malloc(size); assert(vx != NULL);
    vy = (float*)malloc(size); assert(vy != NULL);
    vz = (float*)malloc(size); assert(vz != NULL);

    init(x, y, z, vx, vy, vz, N); /* Init pos / vel data */

    double total_time = 0.0;
    for (int step = 1; step <= nsteps; step++) {
        const double tstart = omp_get_wtime();
        compute_force(x, y, z, vx, vy, vz, DT, N);
        integrate_positions(x, y, z, vx, vy, vz, DT, N);
        const float e = energy(x, y, z, vx, vy, vz, N);
        const double elapsed = omp_get_wtime() - tstart;
        total_time += elapsed;
        printf("Iteration %3d/%3d : energy=%f, %.3f seconds\n", step, nsteps, e, elapsed);
        fflush(stdout);
    }
    const double avg_time = total_time / nsteps;
    printf("Execution time %.3f\n", total_time);
    printf("Average %0.3f Billion Interactions / second\n", 1e-9 * N * N / avg_time);

    free(x);
    free(y);
    free(z);
    free(vx);
    free(vy);
    free(vz);
    return EXIT_SUCCESS;
}
