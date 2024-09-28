/****************************************************************************
 *
 * opencl-denoise.cl -- kernel for opencl-denoise.c
 *
 * Copyright (C) 2021, 2022 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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

void compare_and_swap( __private unsigned char *a,
                       __private unsigned char *b )
{
    if (*a > *b ) {
        unsigned char tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

int IDX(int width, int i, int j)
{
    return (i*width + j);
}

unsigned char median_of_five( __private unsigned char v[5] )
{
    /* We do a partial sort of v[5] using bubble sort until v[2] is
       correctly placed; at the end, v[2] is the median. (There are
       better ways to compute the median-of-five). */
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v  , v+1 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    return v[2];
}

__kernel void
denoise_kernel( __global const unsigned char *bmap,
                __global unsigned char *out,
                int width,
                int height )
{
    const int i = get_global_id(1);
    const int j = get_global_id(0);

    if (i<height && j<width) {
        if ((i>0) && (i<height-1) && (j>0) && (j<width-1)) {
            unsigned char v[5];
            v[0] = bmap[IDX(width, i  , j  )];
            v[1] = bmap[IDX(width, i  , j-1)];
            v[2] = bmap[IDX(width, i  , j+1)];
            v[3] = bmap[IDX(width, i-1, j  )];
            v[4] = bmap[IDX(width, i+1, j  )];

            out[IDX(width, i, j)] = median_of_five(v);
        } else {
            out[IDX(width, i, j)] = bmap[IDX(width, i, j)];
        }
    }
}
