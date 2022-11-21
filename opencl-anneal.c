/****************************************************************************
 *
 * opencl-anneal.c - ANNEAL cellular automaton
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
 cell of a block of size $3 \times 3$](openclanneal1.png)

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

The file [opencl-anneal.c](opencl-anneal.c) contains a serial
implementation of the algorithm that computes the evolution of the
_ANNEAL_ CA after $K$ iterations. The final state is written to a
file. The goal of this exercise is to modify the program to delegate
the computation of new states to the GPU.

Some suggestions:

- Start by developing a version that does _not_ use local
  memory. Transform the `copy_top_bottom()`, `copy_left_right()` and
  `step()` functions into kernels. Note that the size of the
  workgroups that copies the sides of the domain will be different
  from the size of the domain that computes the evolution of the
  automaton (see the following points).

- To copy the ghost cells, use a 1D array of work-items. So, to run
  kernels `copy_top_bottom()` and `copy_left_right()` you need $(N +
  2)$ work-items.

- Since the domain is two-dimensional, it is convenient to organize the
  work-items in two-dimensional blocks. You can set the number of
  elements per side to `SCL_DEFAULT_WG_SIZE2D`, which is a variable
  set automatically during the initialization phase.

- In the `step()` kernel, each work-item computes the new state of a
  coordinate cell $(i, j)$. Remember that you are working on a
  "extended" domain with two more rows and two columns, hence the
  "true" (non-ghost) cells are those with coordinates $1 \leq i, j
  \leq N$. Therefore, each work-item will compute $i, j$ as:
```C.
  const int i = 1 + get_global_id (1);
  const int j = 1 + get_global_id (0);
```
  In this way the work-items will be associated with the coordinate cells
  from $(1, 1)$ onward[^1]. Before making any computation,
  each work-item must verify that $1 \leq i, j \leq N$, so
  that excess work-items are deactivated.

[^1]: OpenCL allows you to launch a kernel by specifying an "offset"
      to be added to the global index of work-items; this allows to
      simplify indexing by avoiding adding 1 explicitly. However, the
      `simpleCL` library does not expose this functionality.

## Using local memory

This program could benefit from the use of local memory, since each
cell is read 9 times by 9 different work-items. However, no
performance improvement will be observed on the lab server, since the
GPUs there have on-board caches that are used automatically. Despite
this, it is useful to use local memory anyway, to see how it can be
done.

Let us assume that a workgroup is a square of size $\mathit{BLKDIM}
\times \mathit{BLKDIM}$ where _BLKDIM_ is an integer multiple of
$N$. Each workgroup copies the elements of the domain portion of its
own competence in a local buffer `buf[BLKDIM+2][BLKDIM+2]` which
includes two ghost rows and columns, and computes the new state of the
cells using the data in the local buffer instead of accessing global
memory.

Here it is useful to use two pairs of indexes $(gi, gj)$ to indicate
the positions of the cells in the global array and $(li, lj)$ for the
cell positions in the local buffer. The idea is that the coordinate
cell $(gi, gj)$ in the global matrix matches the one of coordinates
$(li, lj)$ in the local buffer. Using ghost cell both globally and
locally the calculation of coordinates can be done as follows:

```C.
    const int gi = 1 + get_global_id (1);
    const int gj = 1 + get_global_id (0);
    const int li = 1 + get_local_id (1);
    const int lj = 1 + get_local_id (0);
```

![Figure 3: Copying data from global domain to local storage](openclanneal3.png)

The hardest part is copying the data from the global grid to the local
buffer. Using workgroup of size $\mathit{BLKDIM} \times
\mathit{BLKDIM}$, the copy of the central part (i.e., everything
ùexcluding the hatched area of Figure 3) is carried out with:

```C.
    buf[li][lj] = *IDX(cur, ext_n, gi, gj);
```

where `ext_n = (N + 2)` is the side of the domain including the ghost
area.

![Figure 4: Active work-items while filling the local domain](openclanneal4.png)

To initialize the ghost area you might proceed as follows (Figure 4):

1. The upper and lower ghost area is delegated to the work-items of
   the first row (i.e., those with $li = 1$);

2. The left and right ghost area is delegated to the work-items of the
   first column (i.e., those with $lj = 1$);

3. The ghost area in the corners is delegated to the top left
   work-item with $(li, lj) = (1, 1)$.

(You might be tempted to collapse steps 1 and 2 into a single step
that is carried out, e.g., by the work-items on the first row; this
would be correct, but it would be difficult to generalize the program
to domains whose side is not multiple of $\mathit{BLKDIM}$).

In practice, you will have the following structure:

```C.
    if (li == 1) {
        "fill the cell buf [0] [lj] and buf [BLKDIM + 1] [lj]"
    }
    if (lj == 1) {
        "fill the cell buf [li] [0] and buf [li] [BLKDIM + 1]"
    }
    if (li == 1 && lj == 1) {
        "fill buf [0] [0]"
        "fill buf [0] [BLKDIM + 1]"
        "fill buf [BLKDIM + 1] [0]"
        "fill buf [BLKDIM + 1] [BLKDIM + 1]"
    }
```

Those who want to try an even harder version can try to modify the
code to handle the case where the size of the domain is not an integer
multiple of _BLKDIM_. Deactivating excess work-items it not enough:
you need to modify the initialization of the local memory as well.

To compile without using local storage:

        cc opencl-anneal.c simpleCL.c -o opencl-anneal -lOpenCL

To generate an image at each step:

        cc -DDUMPALL opencl-anneal.c simpleCL.c -o opencl-anneal -lOpenCL

You can edit images in an AVI / MPEG-4 format video with:

        ffmpeg -y -i "opencl-anneal-%06d.pbm" -vcodec mpeg4 opencl-anneal.avi

(the `ffmpeg` command is already installed on the server).

To build the solution by enabling local storage:

        cc -DUSE_LOCAL opencl-anneal.c simpleCL.c -o opencl-anneal-local -lOpenCL

To execute:

        ./opencl-anneal [steps [N]]

Example:

        ./opencl-anneal 64

## Files

- [opencl-anneal.c](opencl-anneal.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpe.h)
- [Animation of the CA](https://youtu.be/UNpl2iUyz3Q)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

typedef unsigned char cell_t;

int IDX(int ext_n, int i, int j)
{
    return (i*ext_n + j);
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
        grid[IDX(ext_n, BOTTOM_GHOST, j)] = grid[IDX(ext_n, TOP, j)]; /* top to bottom halo */
        grid[IDX(ext_n, TOP_GHOST, j)] = grid[IDX(ext_n, BOTTOM, j)]; /* bottom to top halo */
    }
}

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
/* [TODO] This function should be transformed into a kernel */
void copy_left_right(cell_t *grid, int ext_n)
{
    int i;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

    for (i=0; i<ext_n; i++) {
        grid[IDX(ext_n, i, RIGHT_GHOST)] = grid[IDX(ext_n, i, LEFT)]; /* left column to right halo */
        grid[IDX(ext_n, i, LEFT_GHOST)] = grid[IDX(ext_n, i, RIGHT)]; /* right column to left halo */
    }
}

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
                cur[IDX(ext_n, i-1, j-1)] + cur[IDX(ext_n, i-1, j)] + cur[IDX(ext_n, i-1, j+1)] +
                cur[IDX(ext_n, i  , j-1)] + cur[IDX(ext_n, i  , j)] + cur[IDX(ext_n, i  , j+1)] +
                cur[IDX(ext_n, i+1, j-1)] + cur[IDX(ext_n, i+1, j)] + cur[IDX(ext_n, i+1, j+1)];
            next[IDX(ext_n, i, j)] = (nblack >= 6 || nblack == 4);
        }
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

    srand(1234);
    for (i=TOP; i <= BOTTOM; i++) {
        for (j=LEFT; j <= RIGHT; j++) {
            cur[IDX(ext_n, i, j)] = (((float)rand())/RAND_MAX < p);
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

#ifdef SERIAL
    snprintf(fname, sizeof(fname), "cpu-anneal-%06d.pbm", stepno);
#else
    snprintf(fname, sizeof(fname), "opencl-anneal-%06d.pbm", stepno);
#endif

    if ((f = fopen(fname, "w")) == NULL) {
        fprintf(stderr, "Cannot open %s for writing\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(f, "P1\n");
    fprintf(f, "# produced by opencl-anneal.c\n");
    fprintf(f, "%d %d\n", ext_n-2, ext_n-2);
    for (i=1; i<ext_n-1; i++) {
        for (j=1; j<ext_n-1; j++) {
            fprintf(f, "%d ", cur[IDX(ext_n, i, j)]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main( int argc, char* argv[] )
{
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

#ifdef SERIAL
    cell_t *cur = (cell_t*)malloc(ext_size); assert(cur);
    cell_t *next = (cell_t*)malloc(ext_size); assert(next);
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
    sclInitFromFile("opencl-anneal.cl");
    sclKernel copy_top_bottom_kernel = sclCreateKernel("copy_top_bottom_kernel");
    sclKernel copy_left_right_kernel = sclCreateKernel("copy_left_right_kernel");
#ifdef USE_LOCAL
    sclKernel step_kernel = sclCreateKernel("step_kernel_local");
#else
    sclKernel step_kernel = sclCreateKernel("step_kernel");
#endif

#if 0
    /* 1D blocks used for filling ghost area */
    const sclDim copyBlock = DIM1(SCL_DEFAULT_WG_SIZE);
    const sclDim copyGrid = DIM1(sclRoundUp(ext_n, SCL_DEFAULT_WG_SIZE));
    /* 2D blocks used for the update step */
    const sclDim stepBlock = DIM2(SCL_DEFAULT_WG_SIZE2D, SCL_DEFAULT_WG_SIZE2D);
    const sclDim stepGrid = DIM2(sclRoundUp(n, SCL_DEFAULT_WG_SIZE2D),
                                 sclRoundUp(n, SCL_DEFAULT_WG_SIZE2D));
#else
    sclDim copyBlock, copyGrid, stepBlock, stepGrid;
    sclWGSetup1D(ext_n, &copyGrid, &copyBlock);
    sclWGSetup2D(n, n, &stepGrid, &stepBlock);
#endif

    /* Allocate space for host copy of the current grid */
    cell_t *cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    init(cur, ext_n, 0.5);

    /* Allocate space for device copy of |cur| and |next| grids */
    cl_mem d_cur = sclMallocCopy(ext_size, cur, CL_MEM_READ_WRITE);
    cl_mem d_next = sclMalloc(ext_size, CL_MEM_READ_WRITE);

    /* evolve the CA */
    const double tstart = hpc_gettime();
#ifdef DUMPALL
    int dump_at = 1;
#endif
    for (s=0; s<nsteps; s += 1) {
        sclSetArgsEnqueueKernel(copy_top_bottom_kernel,
                                copyGrid, copyBlock,
                                ":b :d",
                                d_cur, ext_n);

        sclSetArgsEnqueueKernel(copy_left_right_kernel,
                                copyGrid, copyBlock,
                                ":b :d",
                                d_cur, ext_n);

        sclSetArgsEnqueueKernel(step_kernel,
                                stepGrid, stepBlock,
                                ":b :b :d",
                                d_cur, d_next, ext_n);

#ifdef DUMPALL
        if (s % dump_at == 0) {
            sclMemcpyDeviceToHost(cur, d_next, ext_size);
            write_pbm(cur, ext_n, s);
        }
        if (s && ((s % (dump_at*1000) == 0)))
            dump_at *= 2;
#endif
        cl_mem d_tmp = d_cur;
        d_cur = d_next;
        d_next = d_tmp;
    }
    sclDeviceSynchronize();
    const double elapsed = hpc_gettime() - tstart;
    /* Copy back result to host */
    sclMemcpyDeviceToHost(cur, d_cur, ext_size);
#endif
    write_pbm(cur, ext_n, s);
    free(cur);
#ifdef SERIAL
    free(next);
#else
    sclFree(d_cur);
    sclFree(d_next);
    sclFinalize();
#endif
    fprintf(stderr, "\n\nElapsed time: %f (%f Mupd/s)\n", elapsed, (n*n/1.0e6)*nsteps/elapsed);
    return EXIT_SUCCESS;
}
