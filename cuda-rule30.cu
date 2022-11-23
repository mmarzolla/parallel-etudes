/****************************************************************************
 *
 * cuda-rule30.cu - "Rule 30" Callular Automaton
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
% HPC - "Rule 30" Cellular Automaton
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-23

The goal of this exercise is to implement the [Rule 30 Cellular
Automaton](https://en.wikipedia.org/wiki/Rule_30) in CUDA.

The Rule 30 CA is a 1D cellular aotmaton that consists of an array
`x[N]` of $N$ integers that can be either 0 or 1. The state of the CA
evolves at discrete time steps: the new state of a cell depends on its
current state, and on the current state of the two neighbors. We
assume cyclic boundary conditions, so that the neighbors of $x[0]$ are
$x[N-1]$ and $x[1]$ and the neighbors of $x[N-1]$ are $x[N-2]$ and
$x[0]$ (Figure 1).

![Figure 1: Rule 30 CA](mpi-rule30-fig1.png)

Given the current values $pqr$ of three adjacent cells, the new value
$q'$ of the middle cell is computed according to Table 1.

:Table 1: Rule 30 (■ = 1, □ = 0):

---------------------------------------- ----- ----- ----- ----- ----- ----- ----- -----
Current configuration $pqr$               ■■■   ■■□   ■□■   ■□□   □■■   □■□   □□■   □□□
New state $q'$ of the central cell         □     □     □     ■     ■     ■     ■     □
---------------------------------------- ----- ----- ----- ----- ----- ----- ----- -----

The sequence □□□■■■■□ = 00011110 on the second row is the binary
representation of decimal 30, from which the name ("Rule 30 CA"); more
details can be found [here](mpi-rule30.pdf).

The file [cuda-rule30.cu](cuda-rule30.cu) contains a serial program
that computes the evolution of the Rule 30 CA, assuming an initial
condition where all cells are 0 except the central one. The program
accepts two optional command line parameters: the domain size $N$ and
the number of steps _nsteps_ to simulate. At the end, the program
produces the image `rule30.pbm` shown in Figure 2 of size $N \times
\textit{nsteps}$.

![Figure 2: Evolution of Rule 30 CA](rule30.png)

Each row of the image represents the state of the automaton at a
specific time step (1 = black, 0 = white). Time moves from top to
bottom: the first line is the initial state (time 0), the second line
is the state at time 1, and so on.

Interestingly, the pattern shown in Figure 2 is similar to the pattern
on the [Conus textile](https://en.wikipedia.org/wiki/Conus_textile)
shell, a highly poisonous marine mollusk which can be found in
tropical seas (Figure 3).

![Figura 3: Conus Textile by Richard Ling - Own work; Location: Cod
Hole, Great Barrier Reef, Australia, CC BY-SA 3.0,
<https://commons.wikimedia.org/w/index.php?curid=293495>](conus-textile.jpg)

The goal of this exercise is to write a parallel version where the
computation of the new states are performed by CUDA threads. In
particular, the `rule30()` function should be turned into a kernel.
Assume that the domain size $N$ is a multiple of the number of threads
per block (_BLKDIM_).

I suggest that you start with a version that does not use shared
memory; this first version should be easily derived from the provided
serial code.

Since each domain cell is read three times by three different threads
within the same block, the computation _might_ benefit from the use of
shared memory.

> **Note:** On modern GPUs the use of shared memory could make very
> little different; it might even make the program _slower_ than the
> version that uses global memory only).  The use of shared memory
> here should be considered as an exercise to see how it can be done.

To use shared memory, refer to the simple example of 1D stencil
computation that we have seen during the class; in this case, the
radius of the stencil is one, i.e., the new state of each cell depends
on the state of a cell and the state of the two neighbors. Be careful,
since in this exercise we are assuming a cyclic domain, whereas in the
stencil computation discussed in the class we did not.

![Figure 3: Using shared memory](cuda-rule30.png)

Looking at Figure 2, you might proceed as follows:

- `d_cur[]` is the current state on GPU memory.

- We create a kernel, say `fill_ghost(...)` that fills the ghost area
  of `d_cur[]`. The kernel will be executed by a single thread only,
  since just two values need to be copied, and therefore will be
  executed as `fill_ghost<<<1, 1>>>(...)`

- We create another kernel that computes the new state of the domain,
  given the current state. To this aim, we use 1D blocks and grid.
  Each block defined a `__shared__` array `buf[BLKDIM+2]`; we need
  `BLKDIM+2` elements since we need to include ghost cells in each
  partition in order to be able to compute the new states of all
  cells.

- Each thread computes the "local" index `lindex` in the `buf[]`
  array, and a "global" index `gindex` in the `d_cur[]` array, of the
  element it is associated with. Care should be taken, since both the
  local and global domains have ghost cells. Therefore, indices should
  be computed as:
```C
      const int lindex = 1 + threadIdx.x;
      const int gindex = 1 + threadIdx.x + blockIdx.x * blockDim.x;
```

- Each thread copies one element from global to shared memory:
```C
      buf[lindex] = cur[gindex];
```

- The first thread of each block also fills the ghost area
  of the shared array `buf[]`:
```C
      if (0 == threadIdx.x) {
          buf[0] = cur[gindex-1];
          buf[BLKDIM + 1] = cur[gindex + BLKDIM];
      }
```

To generate the output image, the new domain should be transferred
back to host memory after each iteration. Then, `d_cur` and `d_next`
must be exchanged before starting the next iteration.

To compile:

        nvcc cuda-rule30.cu -o cuda-rule30

To execute:

        ./cuda-rule30 [width [steps]]

Example:

        ./cuda-rule30 1024 1024

The output is stored to the file `cuda-rule30.pbm`

## Files

- [cuda-rule30.cu](cuda-rule30.cu)
- [hpc.h](hpc.h)

 ***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef unsigned char cell_t;

#ifdef SERIAL
/**
 * Given the current state of the CA, compute the next state.  This
 * version requires that the `cur` and `next` arrays are extended with
 * ghost cells; therefore, `ext_n` is the length of `cur` and `next`
 * _including_ ghost cells.
 *
 *                             +----- ext_n-2
 *                             |   +- ext_n-1
 *   0   1                     V   V
 * +---+-------------------------+---+
 * |///|                         |///|
 * +---+-------------------------+---+
 *
 */
void step( cell_t *cur, cell_t *next, int ext_n )
{
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    int i;
    for (i=LEFT; i<=RIGHT; i++) {
        const cell_t left   = cur[i-1];
        const cell_t center = cur[i  ];
        const cell_t right  = cur[i+1];
        next[i] =
            ( left && !center && !right) ||
            (!left && !center &&  right) ||
            (!left &&  center && !right) ||
            (!left &&  center &&  right);
    }
}
#else
#define BLKDIM 1024

__device__ int d_min(int a, int b)
{
    return (a < b ? a : b);
}

/**
 * Fill ghost cells in device memory. This kernel must be launched
 * with one thread only.
 */
__global__ void fill_ghost( cell_t *cur, int ext_n )
{
    const int LEFT_GHOST = 0;
    const int LEFT = 1;
    const int RIGHT_GHOST = ext_n - 1;
    const int RIGHT = RIGHT_GHOST - 1;
    cur[RIGHT_GHOST] = cur[LEFT];
    cur[LEFT_GHOST] = cur[RIGHT];
}

/**
 * Given the current state `cur` of the CA, compute the `next`
 * state. This function requires that `cur` and `next` are extended
 * with ghost cells; therefore, `ext_n` is the lenght of `cur` and
 * `next` _including_ ghost cells.
 */
__global__ void step( cell_t *cur, cell_t *next, int ext_n )
{
    __shared__ cell_t buf[BLKDIM+2];
    const int gindex = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    const int lindex = 1 + threadIdx.x;

    if ( gindex < ext_n - 1 ) {
        buf[lindex] = cur[gindex];
        if (1 == lindex) {
            /* The thread with threadIdx.x == 0 (therefore, with
               lindex == 1) fills the two ghost cells of `buf[]` (one
               on the left, one on the right). When the width of the
               domain (ext_n - 2) is not multiple of BLKDIM, care must
               be taken. Indeed, if the width is not multiple of
               `BLKDIM`, then the rightmost ghost cell of the last
               thread block is `buf[1+len]`, where len is computed as
               follows: */
            const int len = d_min(BLKDIM, ext_n - 1 - gindex);
            buf[0] = cur[gindex - 1];
            buf[1+len] = cur[gindex + len];
        }

        __syncthreads();

        const cell_t left   = buf[lindex-1];
        const cell_t center = buf[lindex  ];
        const cell_t right  = buf[lindex+1];

        next[gindex] =
            ( left && !center && !right) ||
            (!left && !center &&  right) ||
            (!left &&  center && !right) ||
            (!left &&  center &&  right);
    }
}
#endif

/**
 * Initialize the domain; all cells are 0, with the exception of a
 * single cell in the middle of the domain. `cur` points to an array
 * of length `ext_n`; the length includes two ghost cells.
 */
void init_domain( cell_t *cur, int ext_n )
{
    int i;
    for (i=0; i<ext_n; i++) {
        cur[i] = 0;
    }
    cur[ext_n/2] = 1;
}

/**
 * Dump the current state of the CA to PBM file `out`. `cur` points to
 * an array of length `ext_n` that includes two ghost cells.
 */
void dump_state( FILE *out, const cell_t *cur, int ext_n )
{
    int i;
    const int LEFT = 1;
    const int RIGHT = ext_n - 2;
    for (i=LEFT; i<=RIGHT; i++) {
        fprintf(out, "%d ", cur[i]);
    }
    fprintf(out, "\n");
}

int main( int argc, char* argv[] )
{
    const char *outname = "cuda-rule30.pbm";
    FILE *out;
    int width = 1024, steps = 1024, s;
#ifdef SERIAL
    cell_t *cur, *next;
#else
    cell_t *cur;
    cell_t *d_cur, *d_next;
#endif

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [width [steps]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        width = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        steps = atoi(argv[2]);
    }

    const int ext_width = width + 2;
    const size_t ext_size = ext_width * sizeof(*cur); /* includes ghost cells */
    const int LEFT_GHOST = 0;
    const int LEFT = 1;
    const int RIGHT_GHOST = ext_width - 1;
    const int RIGHT = RIGHT_GHOST - 1;
    /* Create the output file */
    out = fopen(outname, "w");
    if ( !out ) {
        fprintf(stderr, "FATAL: cannot create file \"%s\"\n", outname);
        return EXIT_FAILURE;
    }
    fprintf(out, "P1\n");
    fprintf(out, "# produced by cuda-rule30.cu\n");
    fprintf(out, "%d %d\n", width, steps);

#ifdef SERIAL
    /* Allocate space for the `cur[]` and `next[]` arrays */
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);
    next = (cell_t*)malloc(ext_size); assert(next != NULL);

    /* Initialize the domain */
    init_domain(cur, ext_width);

    /* Evolve the CA */
    for (s=0; s<steps; s++) {

        /* Dump the current state */
        dump_state(out, cur, ext_width);

        /* Fill ghost cells */
        cur[RIGHT_GHOST] = cur[LEFT];
        cur[LEFT_GHOST] = cur[RIGHT];

        /* Compute next state */
        step(cur, next, ext_width);

        /* swap cur and next */
        cell_t *tmp = cur;
        cur = next;
        next = tmp;
    }

    free(cur);
    free(next);
#else
    /* Allocate space for `d_cur[]` and `d_next[]` on the device */
    cudaSafeCall( cudaMalloc((void **)&d_cur, ext_size) );
    cudaSafeCall( cudaMalloc((void **)&d_next, ext_size) );

    /* Allocate space for host copy of `cur[]` */
    cur = (cell_t*)malloc(ext_size); assert(cur != NULL);

    /* Initialize the domain */
    init_domain(cur, ext_width);

    /* Copy input to device */
    cudaSafeCall( cudaMemcpy(d_cur, cur, ext_size, cudaMemcpyHostToDevice) );

    /* Evolve the CA */
    for (s=0; s<steps; s++) {

        /* Dump the current state to the output image */
        dump_state(out, cur, ext_width);

        /* Fill ghost cells */
        fill_ghost<<<1, 1>>>(d_cur, ext_width);
        cudaCheckError();

        /* Compute next state */
        step<<<(width + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_cur, d_next, ext_width);
        cudaCheckError();

        cudaSafeCall( cudaMemcpy(cur, d_next, ext_size, cudaMemcpyDeviceToHost) );

        /* swap d_cur and d_next */
        cell_t *d_tmp = d_cur;
        d_cur = d_next;
        d_next = d_tmp;
    }

    free(cur);
    cudaFree(d_cur);
    cudaFree(d_next);
#endif

    fclose(out);

    return EXIT_SUCCESS;
}
