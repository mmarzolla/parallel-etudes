/****************************************************************************
 *
 * opencl-rule30.cl -- Kernels for opencl-rule30.c
 *
 * Copyright (C) 2021 Moreno Marzolla <https://www.moreno.marzolla.name/>
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

/**
 * Fill ghost cells in device memory. This kernel must be launched
 * with one thread only.
 */
__kernel void
fill_ghost_kernel( __global cell_t *cur,
                   int ext_n )
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
__kernel void
step_kernel( __global const cell_t *cur,
             __global cell_t *next,
             int ext_n )
{
    __local cell_t buf[SCL_DEFAULT_WG_SIZE+2];
    const int gindex = 1 + get_global_id(0);
    const int lindex = 1 + get_local_id(0);
    const int BSIZE = get_local_size(0);

    if ( gindex < ext_n - 1 ) {
        buf[lindex] = cur[gindex];
        if (1 == lindex) {
            /* The thread with threadIdx.x == 0 (therefore, with
               lindex == 1) fills the two ghost cells of `buf[]` (one
               on the left, one on the right). When the width of the
               domain (ext_n - 2) is not multiple of BSIZE, care must
               be taken. Indeed, if the width is not multiple of
               `BSIZE`, then the rightmost ghost cell of the last
               thread block is `buf[1+len]`, where len is computed as
               follows: */
            const int len = min(BSIZE, ext_n - 1 - gindex);
            buf[0] = cur[gindex - 1];
            buf[1+len] = cur[gindex + len];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

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
