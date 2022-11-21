/****************************************************************************
 *
 * cuda-anneal.cu - ANNEAL cellular automaton
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - ANNEAL cellular automaton
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-21

In this exercise we consider a simple two-dimensional, binary Cellular
Automaton called _ANNEAL_ (also known as _twisted majority rule_). The
automaton operates on a square domain of size $N \times N$, where
each cell can have value 0 or 1. Cyclic boundary conditions are
assumed, so that each cell has eight adjacent neighbors. Two cells are
considered adjacent if they share a side or a corner.

The automaton evolves at discrete time steps $t = 0, 1, 2,
\ldots$. The state of a cell at time $t + 1$ depends on its state at
time $t$, and on the state of its neighbors at time $t$. Specifically,
for each cell $x$ let $B_x$ be the number of cells in state 1 within
the neighborhood of size $3 \times 3$ centered on $x$ (including $x$,
so you will always have $0 \leq B_x \leq 9$). If $B_x = 4$ or $B_x
\geq 6$, then the new state of $x$ is 1, otherwise the new state is
0. See Figure 1.

![Figure 1: Examples of computation of the new state of the central
 cell of a block of size $3 \times 3$](cuda-anneal1.png)

To simulate synchrnonous, concurrent updates of all cells, two domains
must be used. The state of a cell is always read from the "current"
domain, and new values are written to the "next" domain. The domains
are exchanged at the end of each step.

The initial state of a cell is chosen at random with equal
probability. Figure 2 shows the evolution of a grid of size $256
\times 256$ after 10, 100 and 1024 iterations. We observe the emergence
of "blobs" of cells that grow over time, with the exception of small
"bubbles" that remain stable. You might be interested in [a short
video showing the evolution of the
automaton](https://youtu.be/UNpl2iUyz3Q) over time.

![Figure 2: Evolution of the _ANNEAL_ automaton ([video](https://youtu.be/UNpl2iUyz3Q))](anneal-demo.png)

The file [cuda-anneal.cu](cuda-anneal.cu) contains a serial
implementation of the algorithm that computes the evolution of the
_ANNEAL_ CA after $K$ iterations. The final state is written to a
file. The goal of this exercise is to modify the program to delegate
the computation of new states to the GPU.

Some suggestions:

- Start by developing a version that does _not_ use shared
  memory. Transform the `copy_top_bottom()`, `copy_left_right()` and
  `step()` functions into kernels. Note that the size of the thread
  block that copies the sides of the domain will be different from the
  size of the domain that computes the evolution of the automaton (see
  the following points).

- To copy the ghost cells, use a 1D array of threads. So, to run
  kernels `copy_top_bottom()` and `copy_left_right()` you need $(N +
  2)$ threads.

- Since the domain is two-dimensional, it is convenient to organize
  the threads in two-dimensional blocksof size $32 \times 32$.

- In the `step()` kernel, each thrad computes the new state of a
  coordinate cell $(i, j)$. Remember that you are working on a
  "extended" domain with two more rows and two columns, hence the
  "true" (non-ghost) cells are those with coordinates $1 \leq i, j \leq N$.
  Therefore, each thread will compute $i, j$ as:
```C
  const int i = 1 + threadIdx.y + blockIdx.y * blockDim.y;
  const int j = 1 + threadIdx.x + blockIdx.x * blockDim.x;
```
  In this way the threads will be associated with the coordinate cells
  from $(1, 1)$ onward. Before making any computation, each
  threa must verify that $1 \leq i, j \leq N$, so that all
  excess threads are deactivated.

## Using local memory

This program might benefit from the use of shared memory, since each
cell is read 9 times by 9 different thrads. However, no performance
improvement is likely to be observed on the server, since the GPUs
there have on-board caches. Despite this, it is useful to use local
memory anyway, to see how it can be done.

Let us assume that thead blocks have size $\mathit{BLKDIM} \times
\mathit{BLKDIM}$ where _BLKDIM_ is a divisor of $N$. Each workgroup
copies the elements of the domain portion of its own competence in a
local buffer `buf[BLKDIM+2][BLKDIM+2]` which includes two ghost rows
and columns, and computes the new state of the cells using the data in
the local buffer instead of accessing global memory.

Here it is useful to use two pairs of indexes $(gi, gj)$ to indicate
the positions of the cells in the global array and $(li, lj)$ for the
cell positions in the local buffer. The idea is that the coordinate
cell $(gi, gj)$ in the global matrix matches the one of coordinates
$(li, lj)$ in the local buffer. Using ghost cell both globally and
locally the calculation of coordinates can be done as follows:

```C
    const int gi = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    const int gj = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    const int li = 1 + threadIdx.y;
    const int lj = 1 + threadIdx.x;
```

![Figure 3: Copying data from global to shared memory](cuda-anneal3.png)

The hardest part is copying the data from the global grid to the
shared buffer. Using blocks of size $\mathit{BLKDIM} \times
\mathit{BLKDIM}$, the copy of the central part (i.e., everything
excluding the hatched area of Figure 3) is carried out with:

```C
    buf[li][lj] = *IDX(cur, ext_n, gi, gj);
```

where `ext_n = (N + 2)` is the side of the domain, including the ghost
area.

![Figure 4: Active threads while filling the shared memory](cuda-anneal4.png)

To initialize the ghost area you might proceed as follows (Figure 4):

1. The upper and lower ghost area is delegated to the threads of the
   first row (i.e., those with $li = 1$);

2. The left and right ghost area is delegated to the threads of the
   first column (i.e., those with $lj = 1$);

3. The corners are delegated to the top left thread with $(li, lj) =
   (1, 1)$.

(You might be tempted to collapse steps 1 and 2 into a single step
that is carried out, e.g., by the threads of the first row; this would
be correct, but it would be difficult to generalize the program to
domains whose sides are not multiple of $\mathit{BLKDIM}$).

In practice, you may use the following schema:

```C
    if ( li == 1 ) {
        "riempi la cella buf[0][lj] e buf[BLKDIM+1][lj]"
    }
    if ( lj == 1 ) {
        "riempi la cella buf[li][0] e buf[li][BLKDIM+1]"
    }
    if ( li == 1 && lj == 1 ) {
        "riempi buf[0][0]"
        "riempi buf[0][BLKDIM+1]"
        "riempi buf[BLKDIM+1][0]"
        "riempi buf[BLKDIM+1][BLKDIM+1]"
    }
```

Those who want to try an even harder version can modify the code to
handle domains whose sides are not multiple of _BLKDIM_. Deactivating
threads outside the domain is not enough: you need to modify the code
that fills the ghost area.

To compile without using shared memory:

        nvcc cuda-anneal.cu -o cuda-anneal

To generate an image after every step:

        nvcc -DDUMPALL cuda-anneal.cu -o cuda-anneal

You can make an AVI / MPEG-4 animatino using:

        ffmpeg -y -i "cuda-anneal-%06d.pbm" -vcodec mpeg4 cuda-anneal.avi

To compile with shared memory:

        nvcc -DUSE_SHARED cuda-anneal.cu -o cuda-anneal-shared

To execute:

        ./cuda-anneal [steps [N]]

Example:

        ./cuda-anneal 64

## Files

- [cuda-anneal.cu](cuda-anneal.cu)
- [hpc.h](hpc.h)
- [Animation of the CA](https://youtu.be/UNpl2iUyz3Q)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifndef SERIAL
/* We use 2D blocks of size (BLKDIM * BLKDIM) to compute
   the next configuration of the automaton */

#define BLKDIM 32

/* We use 1D blocks of (BLKDIM_COPY) threads to copy ghost cells */

#define BLKDIM_COPY 1024
#endif

typedef unsigned char cell_t;

/* The following function makes indexing of the 2D domain
   easier. Instead of writing, e.g., grid[i*ext_n + j] you write
   IDX(grid, ext_n, i, j) to get a pointer to grid[i][j]. This
   function assumes that the size of the CA grid is (ext_n*ext_n),
   where the first and last rows/columns are ghost cells. */
#ifndef SERIAL
__device__ __host__
#endif
cell_t* IDX(cell_t *grid, int ext_n, int i, int j)
{
    return (grid + i*ext_n + j);
}

#ifndef SERIAL
__host__ __device__
#endif
int d_min(int a, int b)
{
    return (a<b ? a : b);
}

/*
  `grid` points to a (ext_n * ext_n) block of bytes; this function
  copies the top and bottom ext_n elements to the opposite halo (see
  figure below).

   LEFT_GHOST=0     RIGHT=ext_n-2
   | LEFT=1         | RIGHT_GHOST=ext_n-1
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
  |Y|YYYYYYYYYYYYYYYY|Y| <- BOTTOM=ext_n - 2
  +-+----------------+-+
  |X|XXXXXXXXXXXXXXXX|X| <- BOTTOM_GHOST=ext_n - 1
  +-+----------------+-+

 */
#ifdef SERIAL
/* [TODO] Transform this function into a kernel */
void copy_top_bottom(cell_t *grid, int ext_n)
{
    int j;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;

    for (j=0; j<ext_n; j++) {
        *IDX(grid, ext_n, BOTTOM_GHOST, j) = *IDX(grid, ext_n, TOP, j); /* top to bottom halo */
        *IDX(grid, ext_n, TOP_GHOST, j) = *IDX(grid, ext_n, BOTTOM, j); /* bottom to top halo */
    }
}
#else
__global__ void copy_top_bottom(cell_t *grid, int ext_n)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;

    if (j < ext_n) {
        *IDX(grid, ext_n, BOTTOM_GHOST, j) = *IDX(grid, ext_n, TOP, j); /* top to bottom halo */
        *IDX(grid, ext_n, TOP_GHOST, j) = *IDX(grid, ext_n, BOTTOM, j); /* bottom to top halo */
    }
}
#endif

/*
  `grid` points to a ext_n*ext_n block of bytes; this function copies
  the left and right ext_n elements to the opposite halo (see figure
  below).

   LEFT_GHOST=0     RIGHT=ext_n-2
   | LEFT=1         | RIGHT_GHOST=ext_n-1
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
  |X|Y              X|Y| <- BOTTOM=ext_n - 2
  +-+----------------+-+
  |X|Y              X|Y| <- BOTTOM_GHOST=ext_n - 1
  +-+----------------+-+

 */
#ifdef SERIAL
/* [TODO] This function should be transformed into a kernel */
void copy_left_right(cell_t *grid, int ext_n)
{
    int i;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

    for (i=0; i<ext_n; i++) {
        *IDX(grid, ext_n, i, RIGHT_GHOST) = *IDX(grid, ext_n, i, LEFT); /* left column to right halo */
        *IDX(grid, ext_n, i, LEFT_GHOST) = *IDX(grid, ext_n, i, RIGHT); /* right column to left halo */
    }
}
#else
__global__ void copy_left_right(cell_t *grid, int ext_n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

    if (i < ext_n) {
        *IDX(grid, ext_n, i, RIGHT_GHOST) = *IDX(grid, ext_n, i, LEFT); /* left column to right halo */
        *IDX(grid, ext_n, i, LEFT_GHOST) = *IDX(grid, ext_n, i, RIGHT); /* right column to left halo */
    }
}
#endif

#ifdef SERIAL
/* Compute the `next` grid given the current configuration `cur`.
   Both grids have (ext_n*ext_n) elements.
   [TODO] This function should be transformed into a kernel. */
void step(cell_t *cur, cell_t *next, int ext_n)
{
    int i, j;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;
    for (i=TOP; i <= BOTTOM; i++) {
        for (j=LEFT; j <= RIGHT; j++) {
            const int nblack =
                *IDX(cur, ext_n, i-1, j-1) + *IDX(cur, ext_n, i-1, j) + *IDX(cur, ext_n, i-1, j+1) +
                *IDX(cur, ext_n, i  , j-1) + *IDX(cur, ext_n, i  , j) + *IDX(cur, ext_n, i  , j+1) +
                *IDX(cur, ext_n, i+1, j-1) + *IDX(cur, ext_n, i+1, j) + *IDX(cur, ext_n, i+1, j+1);
            *IDX(next, ext_n, i, j) = (nblack >= 6 || nblack == 4);
        }
    }
}
#else
/* Compute the next grid given the current configuration. Each grid
   has (ext_n*ext_n) elements. */
__global__ void step(cell_t *cur, cell_t *next, int ext_n)
{
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;
    const int i = TOP + threadIdx.y + blockIdx.y * blockDim.y;
    const int j = LEFT + threadIdx.x + blockIdx.x * blockDim.x;

    if ( i <= BOTTOM && j <= RIGHT ) {
        const int nblack =
            *IDX(cur, ext_n, i-1, j-1) + *IDX(cur, ext_n, i-1, j) + *IDX(cur, ext_n, i-1, j+1) +
            *IDX(cur, ext_n, i  , j-1) + *IDX(cur, ext_n, i  , j) + *IDX(cur, ext_n, i  , j+1) +
            *IDX(cur, ext_n, i+1, j-1) + *IDX(cur, ext_n, i+1, j) + *IDX(cur, ext_n, i+1, j+1);
        *IDX(next, ext_n, i, j) = (nblack >= 6 || nblack == 4);
    }
}

/* Same as above, but using shared memory. This kernel works correctly
   even if the size of the domain is not multiple of BLKDIM.

   Note that, on modern GPUs, this version is actually *slower* than
   the plain version above.  The reason is that neser GPUs have an
   internal cache, and this computation does not reuse data enough to
   pay for the cost of filling the shared memory. */
__global__ void step_shared(cell_t *cur, cell_t *next, int ext_n)
{
    __shared__ cell_t buf[BLKDIM+2][BLKDIM+2];

    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;

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
    const int height = d_min(blockDim.y, ext_n-1-gi);
    const int width  = d_min(blockDim.x, ext_n-1-gj);

    if ( gi <= BOTTOM && gj <= RIGHT ) {
        buf[li][lj] = *IDX(cur, ext_n, gi, gj);
        if (li == 1) {
            /* top and bottom */
            buf[0       ][lj] = *IDX(cur, ext_n, gi-1, gj);
            buf[1+height][lj] = *IDX(cur, ext_n, gi+height, gj);
        }
        if (lj == 1) { /* left and right */
            buf[li][0      ] = *IDX(cur, ext_n, gi, gj-1);
            buf[li][1+width] = *IDX(cur, ext_n, gi, gj+width);
        }
        if (li == 1 && lj == 1) { /* corners */
            buf[0       ][0       ] = *IDX(cur, ext_n, gi-1, gj-1);
            buf[0       ][lj+width] = *IDX(cur, ext_n, gi-1, gj+width);
            buf[1+height][0       ] = *IDX(cur, ext_n, gi+height, gj-1);
            buf[1+height][1+width ] = *IDX(cur, ext_n, gi+height, gj+width);
        }
        __syncthreads(); /* Wait for all threads to fill the shared memory */

        const int nblack =
            buf[li-1][lj-1] + buf[li-1][lj] + buf[li-1][lj+1] +
            buf[li  ][lj-1] + buf[li  ][lj] + buf[li  ][lj+1] +
            buf[li+1][lj-1] + buf[li+1][lj] + buf[li+1][lj+1];
        *IDX(next, ext_n, gi, gj) = (nblack >= 6 || nblack == 4);
    }
}
#endif

/* Initialize the current grid `cur` with alive cells with density
   `p`. */
void init( cell_t *cur, int ext_n, float p )
{
    int i, j;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int TOP = 1;
    const int BOTTOM = ext_n - 2;

    srand(1234); /* initialize PRND */
    for (i=TOP; i <= BOTTOM; i++) {
        for (j=LEFT; j <= RIGHT; j++) {
            *IDX(cur, ext_n, i, j) = (((float)rand())/RAND_MAX < p);
        }
    }
}

/* Write `cur` to a PBM (Portable Bitmap) file whose name is derived
   from the step number `stepno`. */
void write_pbm( cell_t *cur, int ext_n, int stepno )
{
    int i, j;
    char fname[128];
    FILE *f;

    snprintf(fname, sizeof(fname), "cuda-anneal-%06d.pbm", stepno);

    if ((f = fopen(fname, "w")) == NULL) {
        fprintf(stderr, "Cannot open %s for writing\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by cuda-anneal.cu\n");
    fprintf(f, "%d %d\n", ext_n-2, ext_n-2);
    for (i=1; i<ext_n-1; i++) {
        for (j=1; j<ext_n-1; j++) {
            fprintf(f, "%d ", *IDX(cur, ext_n, i, j));
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
    int s, nsteps = 64, n = 512;
    const int MAXN = 2048;

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [nsteps [n]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        nsteps = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        n = atoi(argv[2]);
    }

    if ( n > MAXN ) { /* maximum image size is MAXN */
        fprintf(stderr, "FATAL: the maximum allowed grid size is %d\n", MAXN);
        return EXIT_FAILURE;
    }

    const int ext_n = n + 2;
    const size_t ext_size = ext_n * ext_n * sizeof(cell_t);

    fprintf(stderr, "Anneal CA: steps=%d size=%d\n", nsteps, n);
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
    init(cur, ext_n, 0.5);
    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        copy_top_bottom(cur, ext_n);
        copy_left_right(cur, ext_n);
#ifdef DUMPALL
        write_pbm(cur, ext_n, s);
#endif
        step(cur, next, ext_n);
        cell_t *tmp = cur;
        cur = next;
        next = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;
#else
    /* 1D blocks used for copying sides */
    const dim3 copyBlock(BLKDIM_COPY);
    const dim3 copyGrid((ext_n + BLKDIM_COPY-1)/BLKDIM_COPY);
    /* 2D blocks used for the update step */
    const dim3 stepBlock(BLKDIM, BLKDIM);
    const dim3 stepGrid((n + BLKDIM-1)/BLKDIM, (n + BLKDIM-1)/BLKDIM);

    /* Allocate space for host copy of the current grid */
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    /* Allocate space for device copy of |cur| and |next| grids */
    cudaSafeCall( cudaMalloc((void**)&d_cur, ext_size) );
    cudaSafeCall( cudaMalloc((void**)&d_next, ext_size) );

    init(cur, ext_n, 0.5);
    /* Copy initial grid to device */
    cudaSafeCall( cudaMemcpy(d_cur, cur, ext_size, cudaMemcpyHostToDevice) );

    /* evolve the CA */
    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        copy_top_bottom<<<copyGrid, copyBlock>>>(d_cur, ext_n); cudaCheckError();
        copy_left_right<<<copyGrid, copyBlock>>>(d_cur, ext_n); cudaCheckError();
#ifdef USE_SHARED
        step_shared<<<stepGrid, stepBlock>>>(d_cur, d_next, ext_n); cudaCheckError();
#else
        step<<<stepGrid, stepBlock>>>(d_cur, d_next, ext_n); cudaCheckError();
#endif

#ifdef DUMPALL
        cudaSafeCall( cudaMemcpy(cur, d_next, ext_size, cudaMemcpyDeviceToHost) );
        write_pbm(cur, ext_n, s);
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
    write_pbm(cur, ext_n, s);
    free(cur);
#ifdef SERIAL
    free(next);
#else
    cudaFree(d_cur);
    cudaFree(d_next);
#endif
    fprintf(stderr, "Elapsed time: %f (%f Mupd/s)\n", elapsed, (n*n/1.0e6)*nsteps/elapsed);

    return EXIT_SUCCESS;
}
