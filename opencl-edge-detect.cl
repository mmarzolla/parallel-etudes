/****************************************************************************
 *
 * opencl-edge-detect.cl -- Kernels for opencl-edge-detect.c
 *
 * Copyright (C) 2022 Moreno Marzolla <moreno.marzolla@unibo.it>
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
int IDX(int i, int j, int width)
{
    return (i*width + j);
}

__kernel void
sobel_kernel(__global const unsigned char *in,
             __global unsigned char *edges,
             int width, int height,
             int threshold)
{
    const unsigned char WHITE = 255;
    const unsigned char BLACK = 0;
    const int i = get_global_id(1);
    const int j = get_global_id(0);

    if (i >= height || j >= width)
        return;

    if (i==0 || j==0 || i==height-1 || j==width-1)
        edges[IDX(i, j, width)] = WHITE;
    else {
        /* Compute the gradients Gx and Gy along the x and y
           dimensions */
        const int Gx =
            in[IDX(i-1, j-1, width)] - in[IDX(i-1, j+1, width)]
            + 2*in[IDX(i, j-1, width)] - 2*in[IDX(i, j+1, width)]
            + in[IDX(i+1, j-1, width)] - in[IDX(i+1, j+1, width)];
        const int Gy =
            in[IDX(i-1, j-1, width)] + 2*in[IDX(i-1, j, width)] + in[IDX(i-1, j+1, width)]
            - in[IDX(i+1, j-1, width)] - 2*in[IDX(i+1, j, width)] - in[IDX(i+1, j+1, width)];
        const int magnitude = Gx * Gx + Gy * Gy;
        if  (magnitude > threshold*threshold)
            edges[IDX(i, j, width)] = WHITE;
        else
            edges[IDX(i, j, width)] = BLACK;
    }
}
