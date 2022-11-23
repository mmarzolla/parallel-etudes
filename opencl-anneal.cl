/****************************************************************************
 *
 * opencl-anneal.cl -- Kernels for opencl-anneal.c
 *
 * Copyright 2021, 2022 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
typedef unsigned char cell_t;

int IDX(int width, int i, int j)
{
    return (i*width + j);
}

__kernel void
copy_top_bottom_kernel(__global cell_t *grid,
                       int ext_width,
                       int ext_height)
{
    const int j = get_global_id(0);
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int TOP_GHOST = TOP - 1;
    const int BOTTOM_GHOST = BOTTOM + 1;

    if (j < ext_width) {
        grid[IDX(ext_width, BOTTOM_GHOST, j)] = grid[IDX(ext_width, TOP, j)]; /* top to bottom halo */
        grid[IDX(ext_width, TOP_GHOST, j)] = grid[IDX(ext_width, BOTTOM, j)]; /* bottom to top halo */
    }
}

__kernel void
copy_left_right_kernel(__global cell_t *grid,
                       int ext_width,
                       int ext_height)
{
    const int i = get_global_id(0);
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int LEFT_GHOST = LEFT - 1;
    const int RIGHT_GHOST = RIGHT + 1;

    if (i < ext_height) {
        grid[IDX(ext_width, i, RIGHT_GHOST)] = grid[IDX(ext_width, i, LEFT)]; /* left column to right halo */
        grid[IDX(ext_width, i, LEFT_GHOST)] = grid[IDX(ext_width, i, RIGHT)]; /* right column to left halo */
    }
}

__kernel void
step_kernel(__global const cell_t *cur,
            __global cell_t *next,
            int ext_width,
            int ext_height)
{
    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;
    const int i = TOP + get_global_id(1);
    const int j = LEFT + get_global_id(0);

    if ( i <= BOTTOM && j <= RIGHT ) {
        int nblack = 0;
#pragma unroll
        for (int di=-1; di<=1; di++) {
#pragma unroll
            for (int dj=-1; dj<=1; dj++) {
                nblack += cur[IDX(ext_width, i+di, j+dj)];
            }
        }
        next[IDX(ext_width, i, j)] = (nblack >= 6 || nblack == 4);
    }
}

/* Same as above, but using local memory. This kernel works correctly
   even if the size of the domain is not multiple of BLKDIM.

   Note that, on modern GPUs, this version is actually *slower* than
   the plain version above.  The reason is that neser GPUs have an
   internal cache, and this computation does not reuse data enough to
   pay for the cost of filling the local memory. */
__kernel void
step_kernel_local(__global const cell_t *cur,
                  __global cell_t *next,
                  int ext_width,
                  int ext_height)
{
    __local cell_t buf[SCL_DEFAULT_WG_SIZE2D+2][SCL_DEFAULT_WG_SIZE2D+2];

    const int LEFT = 1;
    const int RIGHT = ext_width - 2;
    const int TOP = 1;
    const int BOTTOM = ext_height - 2;

    /* "global" indexes */
    const int gi = TOP + get_global_id(1);
    const int gj = LEFT + get_global_id(0);
    /* "local" indexes */
    const int li = 1 + get_local_id(1);
    const int lj = 1 + get_local_id(0);

    /* The following variables are needed to handle the case of a
       domain whose size is not multiple of BLKDIM.

       eight and width of the (NOT extended) subdomain handled by
       this thread block. Its maximum size is blockdim.x * blockDim.y,
       but could be less than that if the domain size is not a
       multiple of the block size. */
    const int height = min((int)get_local_size(1), ext_height-1-gi);
    const int width  = min((int)get_local_size(0), ext_width-1-gj);

    if ( gi <= BOTTOM && gj <= RIGHT ) {
        buf[li][lj] = cur[IDX(ext_width, gi, gj)];
        if (li == 1) {
            /* top and bottom */
            buf[0       ][lj] = cur[IDX(ext_width, gi-1, gj)];
            buf[1+height][lj] = cur[IDX(ext_width, gi+height, gj)];
        }
        if (lj == 1) { /* left and right */
            buf[li][0      ] = cur[IDX(ext_width, gi, gj-1)];
            buf[li][1+width] = cur[IDX(ext_width, gi, gj+width)];
        }
        if (li == 1 && lj == 1) { /* corners */
            buf[0       ][0       ] = cur[IDX(ext_width, gi-1, gj-1)];
            buf[0       ][lj+width] = cur[IDX(ext_width, gi-1, gj+width)];
            buf[1+height][0       ] = cur[IDX(ext_width, gi+height, gj-1)];
            buf[1+height][1+width ] = cur[IDX(ext_width, gi+height, gj+width)];
        }
        barrier(CLK_LOCAL_MEM_FENCE); /* Wait for all threads to fill the local memory */

        int nblack = 0;
#pragma unroll
        for (int di=-1; di<=1; di++) {
#pragma unroll
            for (int dj=-1; dj<=1; dj++) {
                nblack += buf[li+di][lj+dj];
            }
        }
        next[IDX(ext_width, gi, gj)] = (nblack >= 6 || nblack == 4);
    }
}
