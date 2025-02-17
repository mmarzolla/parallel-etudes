/****************************************************************************
 *
 * cuda-anneal.cu - ANNEAL cellular automaton
 *
 * Copyright (C) 2017--2024 Moreno Marzolla <https://www.unibo.it/sitoweb/moreno.marzolla/>
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
% HPC - ANNEAL cellular automaton
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla/)
% Last updated: 2024-11-21

The ANNEAL Callular Automaton (also known as _twisted majority rule_)
is a simple two-dimensional, binary CA defined on a grid of size $W
\times H$. Cyclic boundary conditions are assumed, so that each cell
has eight neighbors. Two cells are adjacent if they share a side or a
corner.

The automaton evolves at discrete time steps $0, 1, \ldots$. The new
state $x'$ at time $t+1$ of a cell $x$ depends on its current state at
time $t$ and on the current state of its neighbors. Specifically, let
$B_x$ be the number of cells in state 1 within the neighborhood of
size $3 \times 3$ centered on $x$ (including $x$ itself). Then, if
$B_x = 4$ or $B_x \geq 6$ the new state $x'$ is 1, otherwise it is 0:

$$
x' = \begin{cases}
1 & \mbox{if $B_x=4$ or $B_x \geq 6$} \\
0 & \mbox{otherwise}
\end{cases}
$$

See Figure 1 for some examples.

![Figure 1: Computation of the new state of the central cell of a
 block of size $3 \times 3$](cuda-anneal1.svg)

To simulate synchronous, concurrent updates of all cells, two domains
must be used. The state of a cell is read from the "current" domain,
and new values are written to the "next" domain. "Current" and "next"
are exchanged at the end of each step.

The initial states are chosen at random with uniform
probability. Figure 2 shows the evolution of a grid of size $256
\times 256$ after 10, 100 and 1024 steps. We observe the emergence of
"blobs" that grow over time, with the exception of small "specks".

![Figure 2: Evolution of the _ANNEAL_ CA](anneal-demo.png)

I made a short YouTube video to show the evolution of the automaton
over time:

<iframe style="display:block; margin:0 auto" width="560" height="315" src="https://www.youtube-nocookie.com/embed/TSHWSjICCxs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The program [cuda-anneal.cu](cuda-anneal.cu) computes the evolution of
the _ANNEAL_ CA after $K$ iterations. The final state is written to a
file. The goal of this exercise is to modify the program to use the
GPU to update the domain.

Some suggestions:

- Start writing a version that does _not_ use shared memory. Transform
  `copy_top_bottom()`, `copy_left_right()` and `step()` into
  kernels. The size of the thread blocks that copy the sides of the
  domain will be different from the size of the domain that computes
  the evolution of the automaton (see below).

- Use a 1D array of threads to copy the ghost cells.
  `copy_top_bottom()` requires $(W + 2)$ threads, and
  `copy_left_right()` requires $(H + 2)$ threads (Figure 3).

- To compute new states, create at least $W \times H$ threads
  organized in 2D blocks of size $32 \times 32$.

- In the `step()` kernel, each thread computes the new state of a cell
  $(i, j)$. We are working with an extended domain with ghost
  rows/columns, hence the "true" (non-ghost) cells are those with
  coordinates $1 \leq i \leq H$, $1 \leq j \leq W$.  Therefore, each
  thread will compute $i, j$ as:
```C
  const int i = 1 + threadIdx.y + blockIdx.y * blockDim.y;
  const int j = 1 + threadIdx.x + blockIdx.x * blockDim.x;
```
  Before making any computation, each thread must verify that $1 \leq i
  \leq H$, $1 \leq j \leq W$, so that excess threads are
  deactivated.

![Figure 3: Initializing the ghost area](cuda-anneal5.svg)

## Using local memory

This program _might_ benefit from the use of shared memory, since each
cell is read 9 times by 9 different threads in the same
block. However, you might observe no improvement on modern GPUs, since
they have on-board caches. In fact, you might actually observe that
using shared memory _decreases_ the performance of the computation,
since there is not enough data reuse to compensate for the overhead of
managing the shared memory.  Despite this, it is a useful exercise to
use local memory anyway, to see how it can be done.

Let us assume that thead blocks have size $\mathit{BLKDIM} \times
\mathit{BLKDIM}$, where _BLKDIM = 32_ divides $W$ and $H$. Each block
uses a local buffer `buf[BLKDIM+2][BLKDIM+2]` which includes a ghost
area. The buffer is filled with data from the current domain, and the
new cell states are computed using the data from the buffer.

Each thread is mapped to cell $(gi, gj)$ in the global domain, and to
a copy of the same cell at coordinates $(li, lj)$ in the local
buffer. The coordinates can be computes as follows:

```C
    const int gi = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    const int gj = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    const int li = 1 + threadIdx.y;
    const int lj = 1 + threadIdx.x;
```

![Figure 4: Copying data from global to shared memory](cuda-anneal3.svg)

There are several ways to fill the ghost area, all of them rather
cumbersome and potentially inefficient. The solution proposed below is
one of them; other possibilities exist.

We use blocks of size $\mathit{BLKDIM} \times
\mathit{BLKDIM}$. Filling the central part of the local domain (i.e.,
everything excluding the ghost area) is done with a single instruction
executed by all threads:

```C
    buf[li][lj] = *IDX(cur, ext_width, gi, gj);
```

where `ext_width = (W + 2)` is the width of the domain including the
ghost area.

![Figure 5: Active threads while filling the shared memory](cuda-anneal4.svg)

Filling the ghost area can be done as follows (see Figure 5):

1. The upper and lower rows are delegated to the threads of the first
   row (i.e., those with $li = 1$);

2. The left and right columns are delegated to the threads of the
   first column (i.e., those with $lj = 1$);

3. The corners are delegated to the top left thread with $(li, lj) =
   (1, 1)$.

You might be tempted to collapse steps 1 and 2 into a single step that
is carried out, e.g., by the threads of the first row. This could
work, but it would be difficult to generalize the program to domains
whose sides $W, H$ are not multiple of _BLKDIM_.

In practice, you may use the following schema:

```C
    if ( li == 1 ) {
        "fill buf[0][lj] and buf[BLKDIM+1][lj]"
    }
    if ( lj == 1 ) {
        "fill buf[li][0] and buf[li][BLKDIM+1]"
    }
    if ( li == 1 && lj == 1 ) {
        "fill buf[0][0]"
        "fill buf[0][BLKDIM+1]"
        "fill buf[BLKDIM+1][0]"
        "fill buf[BLKDIM+1][BLKDIM+1]"
    }
```

Handling the ghost area is more difficult if $W, H$ are not multiple
of _BLKDIM_. Deactivating threads outside the domain is not enough:
you need to modify the code that fills the ghost area.

To compile without using shared memory:

        nvcc cuda-anneal.cu -o cuda-anneal

To generate an image after every step:

        nvcc -DDUMPALL cuda-anneal.cu -o cuda-anneal

You can make an AVI / MPEG-4 animation using:

        ffmpeg -y -i "cuda-anneal-%06d.pbm" -vcodec mpeg4 cuda-anneal.avi

To compile with shared memory:

        nvcc --ptxas-options=-v -DUSE_SHARED cuda-anneal.cu -o cuda-anneal-shared

To execute:

        ./cuda-anneal [steps [W [H]]]

Example:

        ./cuda-anneal 64

## References

- Tommaso Toffoli, Norman Margolus, _Cellular Automata Machines: a new
  environment for modeling_, MIT Press, 1987, ISBN 9780262526319.
  [PDF](https://people.csail.mit.edu/nhm/cam-book.pdf) from Norman
  Margolus home page.

## Files

- [cuda-anneal.cu](cuda-anneal.cu)
- [hpc.h](hpc.h)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

#ifndef SERIAL
/* We use 2D blocks of size (BLKDIM * BLKDIM) to compute
   the next configuration of the automaton */

#define BLKDIM 32

/* We use 1D blocks of (BLKDIM_COPY) threads to copy ghost cells */

#define BLKDIM_COPY 1024
#endif

typedef unsigned char cell_t;

/* The following function simplifies indexing of the 2D
   domain. Instead of writing grid[i*ext_width + j] you write
   IDX(grid, ext_width, i, j) to get a pointer to grid[i][j]. This
   function assumes that the size of the CA grid is
   (ext_width*ext_height), where the first and last rows/columns are
   ghost cells. */
#ifndef SERIAL
__device__ __host__
#endif
cell_t* IDX(cell_t *grid, int ext_width, int i, int j)
{
    return (grid + i*ext_width + j);
}

/*
  `grid` points to a (ext_width * ext_height) block of bytes; this
  function copies the top and bottom ext_width elements to the
  opposite halo (see figure below).

   LEFT_GHOST=0     RIGHT=ext_width-2
   | LEFT=1         | RIGHT_GHOST=ext_width-1
   | |              | |
   v v              v v
  +-+----------------+-+
  |Y|YYYYYYYYYYYYYYYY|Y| <- TOP_GHOST=0
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- TOP=1
  | |                | |
  | |                | |
  | |                | |
  | |                | |
  |Y|YYYYYYYYYYYYYYYY|Y| <- BOTTOM=ext_height - 2
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- BOTTOM_GHOST=ext_height - 1
  +-+----------------+-+

 */
#ifdef SERIAL
/* [TODO] Transform this function into a kernel */
void copy_top_bottom(cell_t *grid, int ext_width, int ext_height)
{
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;

    for (int j=0; j<ext_width; j++) {
        *IDX(grid, ext_width, BOTTOM_GHOST, j) = *IDX(grid, ext_width, TOP, j); /* top to bottom halo */
        *IDX(grid, ext_width, TOP_GHOST, j) = *IDX(grid, ext_width, BOTTOM, j); /* bottom to top halo */
    }
}
#else
__global__ void copy_top_bottom(cell_t *grid, int ext_width, int ext_height)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;

    if (j < ext_width) {
        *IDX(grid, ext_width, BOTTOM_GHOST, j) = *IDX(grid, ext_width, TOP, j); /* top to bottom halo */
        *IDX(grid, ext_width, TOP_GHOST, j) = *IDX(grid, ext_width, BOTTOM, j); /* bottom to top halo */
    }
}
#endif

/*
  `grid` points to a ext_width*ext_height block of bytes; this
  function copies the left and right ext_height elements to the
  opposite halo (see figure below).

   LEFT_GHOST=0     RIGHT=ext_width-2
   | LEFT=1         | RIGHT_GHOST=ext_width-1
   | |              | |
   v v              v v
  +-+----------------+-+
  |X|Y              X|Y| <- TOP_GHOST=0
  +-+----------------+-+
  |X|Y              X|Y| <- TOP=1
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y|
  |X|Y              X|Y| <- BOTTOM=ext_height - 2
  +-+----------------+-+
  |X|Y              X|Y| <- BOTTOM_GHOST=ext_height - 1
  +-+----------------+-+

 */
#ifdef SERIAL
/* [TODO] This function should be transformed into a kernel */
void copy_left_right(cell_t *grid, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

    for (int i=0; i<ext_height; i++) {
        *IDX(grid, ext_width, i, RIGHT_GHOST) = *IDX(grid, ext_width, i, LEFT); /* left column to right halo */
        *IDX(grid, ext_width, i, LEFT_GHOST) = *IDX(grid, ext_width, i, RIGHT); /* right column to left halo */
    }
}
#else
__global__ void copy_left_right(cell_t *grid, int ext_width, int ext_height)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

    if (i < ext_height) {
        *IDX(grid, ext_width, i, RIGHT_GHOST) = *IDX(grid, ext_width, i, LEFT); /* left column to right halo */
        *IDX(grid, ext_width, i, LEFT_GHOST) = *IDX(grid, ext_width, i, RIGHT); /* right column to left halo */
    }
}
#endif

#ifdef SERIAL
/* Compute the `next` grid given the current configuration `cur`.
   Both grids have (ext_width * ext_height) elements.

   [TODO] This function should be transformed into a kernel. */
void step(cell_t *cur, cell_t *next, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    for (int i=TOP; i <= BOTTOM; i++) {
        for (int j=LEFT; j <= RIGHT; j++) {
            int nblack = 0;
            for (int di=-1; di<=1; di++) {
                for (int dj=-1; dj<=1; dj++) {
                    nblack += *IDX(cur, ext_width, i+di, j+dj);
                }
            }
            *IDX(next, ext_width, i, j) = (nblack >= 6 || nblack == 4);
        }
    }
}
#else
__device__
int d_min(int a, int b)
{
    return (a<b ? a : b);
}

/* Compute the next grid given the current configuration. Each grid
   has (ext_width * ext_height) elements. */
__global__ void step(cell_t *cur, cell_t *next, int ext_width, int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int i = TOP + threadIdx.y + blockIdx.y * blockDim.y;
    const int j = LEFT + threadIdx.x + blockIdx.x * blockDim.x;

    if ( i <= BOTTOM && j <= RIGHT ) {
        int nblack = 0;
        /* The `#pragma unroll` directive instructs nvcc to unroll the
           "for" loop immediately following it. Loop unrolling is a
           well-known optimization techniques that consists of
           replacing, e.g., the code:

           for (int i=0; i<5; i++) {
              foo(i);
           }

           with

           foo(0);
           foo(1);
           foo(2);
           foo(3);
           foo(4);

           to avoid the overhead of testing, branching and
           incrementing the loop counter.

           Loop unrolloing is normally left to the compiler; however,
           nvcc allows the user to explicitly ask for unrolling
           certain loop(s). Unrolling is useful for "small" loops with
           a simple body that is iterated only a few times (ideally,
           the number of iterations should be known at compile-time).
        */
#pragma unroll
        for (int di=-1; di<=1; di++) {
#pragma unroll
            for (int dj=-1; dj<=1; dj++) {
                nblack += *IDX(cur, ext_width, i+di, j+dj);
            }
        }
        *IDX(next, ext_width, i, j) = (nblack >= 6 || nblack == 4);
    }
}

/* Same as above, but using shared memory. This kernel works correctly
   even if the size of the domain is not multiple of BLKDIM.

   Note that, on modern GPUs, this version is actually *slower* than
   the plain version above.  The reason is that neser GPUs have an
   internal cache, and this computation does not reuse data enough to
   pay for the cost of filling the shared memory. */
__global__ void step_shared(cell_t *cur, cell_t *next, int ext_width, int ext_height)
{
    __shared__ cell_t buf[BLKDIM+2][BLKDIM+2];

    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;

    /* "global" indexes */
    const int gi = TOP + threadIdx.y + blockIdx.y * blockDim.y;
    const int gj = LEFT + threadIdx.x + blockIdx.x * blockDim.x;
    /* "local" indexes */
    const int li = 1 + threadIdx.y;
    const int lj = 1 + threadIdx.x;

    /* The following variables are needed to handle the case of a
       domain whose size is not multiple of BLKDIM.

       height and width of the (NOT extended) subdomain handled by
       this thread block. Its maximum size is blockdim.x * blockDim.y,
       but could be less than that if the domain size is not a
       multiple of the block size. */
    const int height = d_min(blockDim.y, ext_height-1-gi);
    const int width  = d_min(blockDim.x, ext_width-1-gj);

    if ( gi <= BOTTOM && gj <= RIGHT ) {
        buf[li][lj] = *IDX(cur, ext_width, gi, gj);
        if (li == 1) {
            /* top and bottom */
            buf[0       ][lj] = *IDX(cur, ext_width, gi-1, gj);
            buf[1+height][lj] = *IDX(cur, ext_width, gi+height, gj);
        }
        if (lj == 1) { /* left and right */
            buf[li][0      ] = *IDX(cur, ext_width, gi, gj-1);
            buf[li][1+width] = *IDX(cur, ext_width, gi, gj+width);
        }
        if (li == 1 && lj == 1) { /* corners */
            buf[0       ][0       ] = *IDX(cur, ext_width, gi-1, gj-1);
            buf[0       ][lj+width] = *IDX(cur, ext_width, gi-1, gj+width);
            buf[1+height][0       ] = *IDX(cur, ext_width, gi+height, gj-1);
            buf[1+height][1+width ] = *IDX(cur, ext_width, gi+height, gj+width);
        }
        __syncthreads(); /* Wait for all threads to fill the shared memory */

        int nblack = 0;
#pragma unroll
        for (int di=-1; di<=1; di++) {
#pragma unroll
            for (int dj=-1; dj<=1; dj++) {
                nblack += buf[li+di][lj+dj];
            }
        }
        *IDX(next, ext_width, gi, gj) = (nblack >= 6 || nblack == 4);
    }
}
#endif

/* Initialize the current grid `cur` with alive cells with density
   `p`. */
void init( cell_t *cur, int ext_width, int ext_height, float p )
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;

    srand(1234); /* initialize PRND */
    for (int i=TOP; i <= BOTTOM; i++) {
        for (int j=LEFT; j <= RIGHT; j++) {
            *IDX(cur, ext_width, i, j) = (((float)rand())/RAND_MAX < p);
        }
    }
}

/* Write `cur` to a PBM (Portable Bitmap) file whose name is derived
   from the step number `stepno`. */
void write_pbm( cell_t *cur, int ext_width, int ext_height, int stepno )
{
    char fname[128];
    FILE *f;
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;

    snprintf(fname, sizeof(fname), "cuda-anneal-%06d.pbm", stepno);

    if ((f = fopen(fname, "w")) == NULL) {
        fprintf(stderr, "Cannot open %s for writing\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by cuda-anneal.cu\n");
    fprintf(f, "%d %d\n", ext_width-2, ext_height-2);
    for (int i=LEFT; i<=RIGHT; i++) {
        for (int j=TOP; j<=BOTTOM; j++) {
            fprintf(f, "%d ", *IDX(cur, ext_width, i, j));
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main( int argc, char* argv[] )
{
#ifdef SERIAL
    cell_t *cur, *next;
#else
    cell_t *cur;
    cell_t *d_cur, *d_next;
#endif
    int nsteps = 64, width = 512, height = 512, s;
    const int MAXN = 2048;

    if ( argc > 4 ) {
        fprintf(stderr, "Usage: %s [nsteps [W [H]]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        nsteps = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        width = height = atoi(argv[2]);
    }

    if ( argc > 3 ) {
        height = atoi(argv[3]);
    }

    if ( width > MAXN || height > MAXN ) { /* maximum image size is MAXN */
        fprintf(stderr, "FATAL: the maximum allowed grid size is %d\n", MAXN);
        return EXIT_FAILURE;
    }

    const int ext_width = width + 2;
    const int ext_height = height + 2;
    const size_t ext_size = ext_width * ext_height * sizeof(cell_t);

    fprintf(stderr, "Anneal CA: steps=%d size=%d x %d\n", nsteps, width, height);
#ifndef SERIAL
#ifdef USE_SHARED
    printf("Using shared memory\n");
#else
    printf("NOT using shared memory\n");
#endif
#endif

#ifdef SERIAL
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    next = (cell_t*)malloc(ext_size); assert(next != NULL);
    init(cur, ext_width, ext_height, 0.5);
    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        copy_top_bottom(cur, ext_width, ext_height);
        copy_left_right(cur, ext_width, ext_height);
#ifdef DUMPALL
        write_pbm(cur, ext_width, ext_height, s);
#endif
        step(cur, next, ext_width, ext_height);
        cell_t *tmp = cur;
        cur = next;
        next = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;
#else
    /* 1D blocks used for copying sides */
    const dim3 copyLRBlock(BLKDIM_COPY);
    const dim3 copyLRGrid((ext_height + BLKDIM_COPY-1)/BLKDIM_COPY);
    const dim3 copyTBBlock(BLKDIM_COPY);
    const dim3 copyTBGrid((ext_width + BLKDIM_COPY-1)/BLKDIM_COPY);
    /* 2D blocks used for the update step */
    const dim3 stepBlock(BLKDIM, BLKDIM);
    const dim3 stepGrid((width + BLKDIM-1)/BLKDIM, (height + BLKDIM-1)/BLKDIM);

    /* Allocate space for host copy of the current grid */
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    /* Allocate space for device copy of |cur| and |next| grids */
    cudaSafeCall( cudaMalloc((void**)&d_cur, ext_size) );
    cudaSafeCall( cudaMalloc((void**)&d_next, ext_size) );

    init(cur, ext_width, ext_height, 0.5);
    /* Copy initial grid to device */
    cudaSafeCall( cudaMemcpy(d_cur, cur, ext_size, cudaMemcpyHostToDevice) );

    /* evolve the CA */
    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        copy_top_bottom<<<copyTBGrid, copyTBBlock>>>(d_cur, ext_width, ext_height); cudaCheckError();
        copy_left_right<<<copyLRGrid, copyLRBlock>>>(d_cur, ext_width, ext_height); cudaCheckError();
#ifdef USE_SHARED
        step_shared<<<stepGrid, stepBlock>>>(d_cur, d_next, ext_width, ext_height); cudaCheckError();
#else
        step<<<stepGrid, stepBlock>>>(d_cur, d_next, ext_width, ext_height); cudaCheckError();
#endif

#ifdef DUMPALL
        cudaSafeCall( cudaMemcpy(cur, d_next, ext_size, cudaMemcpyDeviceToHost) );
        write_pbm(cur, ext_width, ext_height, s);
#endif
        cell_t *d_tmp = d_cur;
        d_cur = d_next;
        d_next = d_tmp;
    }
    cudaDeviceSynchronize();
    const double elapsed = hpc_gettime() - tstart;
    /* Copy back result to host */
    cudaSafeCall( cudaMemcpy(cur, d_cur, ext_size, cudaMemcpyDeviceToHost) );
#endif
    write_pbm(cur, ext_width, ext_height, s);
    free(cur);
#ifdef SERIAL
    free(next);
#else
    cudaFree(d_cur);
    cudaFree(d_next);
#endif
    fprintf(stderr, "Elapsed time: %f (%f Mops/s)\n", elapsed, (width*height/1.0e6)*nsteps/elapsed);

    return EXIT_SUCCESS;
}
