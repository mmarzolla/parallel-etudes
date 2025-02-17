/****************************************************************************
 *
 * opencl-edge-detect.cl - Kernels for opencl-edge-detect.c
 *
 * Copyright (C) 2022 Moreno Marzolla
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
