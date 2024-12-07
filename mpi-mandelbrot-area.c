/****************************************************************************
 *
 * mpi-mandelbrot-area.c - Area of the Mandelbrot set
 *
 * Copyright (C) 2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Area of the Mandelbrot set
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-12-07

The Mandelbrot set is the set of black points in Figure 1.

![Figure 1: The Mandelbrot set](mandelbrot-set.png)

The file [mpi-mandelbrot-area.c](mpi-mandelbrot-area.c) contains a
serial program that computes an estimate of the area of the Mandelbrot
set.

The program works as follows. First, we identify a rectangle in the
complex plane that contains the Mandelbrot set. Let _(XMIN, YMIN)_ and
_(XMAX, YMAX)_ be the upper left and lower right coordinates of such a
rectangle (the program defines these values).

The program overlaps a regular grid of $N \times N$ points over the
bounding rectangle. For each point we decide whether it belongs to the
Mandelbrot set. Let $x$ be the number of points that belong to the
Mandelbrot set (by construction, $x \leq N \times N$). Let $B$ be the
area of the bounding rectangle defined as

$$
B := (\mathrm{XMAX} - \mathrm{XMIN}) \times (\mathrm{YMAX} - \mathrm{YMIN})
$$

Then, the area $A$ of the Mandelbrot set can be approximated as

$$
A \approx \frac{x}{N^2} \times B
$$

The approximation gets better as the number of points $N$ becomes
larger. The exact value of $A$ is not known, but there are [some
estimates](https://www.fractalus.com/kerry/articles/area/mandelbrot-area.html).

Modify the serial program to use the distributed-memory parallelism
provided by MPI. To this aim, you can logically distribute the $N$
rows of the grid across $P$ MPI processes. No communication is
required, since each process can compute the start (included) and end
(excluded) line with the usual formulas:

```
const int local_ystart = (N * my_rank) / comm_sz;
const int local_yend = (N * (my_rank+1)) / comm_sz;
```

Once each process has computed the number of points of the Mandelbrot
set in its assigned strip, all processes can compute a sum-reduction
of the local values to get the result.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-mandelbrot-area.c -o mpi-mandelbrot-area -lm

To execute:

        mpirun -n NPROC ./mpi-mandelbrot-area [N]

Example:

        mpirun -n 4 ./mpi-mandelbrot-area 1000

## Files

- [mpi-mandelbrot-area.c](mpi-mandelbrot-area.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

/* Higher value = slower to detect points that belong to the Mandelbrot set */
const int MAXIT = 10000;

/* We consider the region on the complex plane -2.25 <= Re <= 0.75
   -1.4 <= Im <= 1.5 */
const float XMIN = -2.25, XMAX = 0.75;
const float YMIN = -1.4, YMAX = 1.5;

/**
 * Performs the iteration z = z*z+c, until ||z|| > 2 when point is
 * known to be outside the Mandelbrot set. Return the number of
 * iterations until ||z|| > 2, or MAXIT.
 */
int iterate(float cx, float cy)
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

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    int N = 1000;
    int ninside = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        N = atoi(argv[1]);
    }

    const double tstart = MPI_Wtime();
#ifdef SERIAL
    if ( 0 == my_rank ) {
        /* [TODO] This is not a true parallel version, since the master
           does everything */
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                const float cx = XMIN + (XMAX-XMIN)*j/N;
                const float cy = YMIN + (YMAX-YMIN)*i/N;
                const int it = iterate(cx, cy);
                ninside += (it >= MAXIT);
            }
        }
    }
#else
    const int local_ystart = N * my_rank / comm_sz;
    const int local_yend = (N * (my_rank + 1)) / comm_sz;
    int local_ninside = 0;
    for (int i=local_ystart; i<local_yend; i++) {
        for (int j=0; j<N; j++) {
            const float cx = XMIN + (XMAX-XMIN)*j/N;
            const float cy = YMIN + (YMAX-YMIN)*i/N;
            const int it = iterate(cx, cy);
            local_ninside += (it >= MAXIT);
        }
    }
    printf("Rank=%d local_ninside=%d\n", my_rank, local_ninside);
    MPI_Reduce(&local_ninside,  /* sendbuf      */
               &ninside,        /* recfbuf      */
               1,               /* count        */
               MPI_INT,         /* datatype     */
               MPI_SUM,         /* op           */
               0,               /* root         */
               MPI_COMM_WORLD);
#endif
    const double elapsed = MPI_Wtime() - tstart;

    if (0 == my_rank) {
        printf("N = %d, ninside = %u\n", N, ninside);

        /* Compute area and error estimate and output the results */
        const double area = (XMAX-XMIN)*(YMAX-YMIN)*ninside/(N*N);

        printf("Area of Mandlebrot set = %f\n", area);
        printf("Correct answer should be around 1.50659\n");
        printf("Elapsed time: %f\n", elapsed);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
