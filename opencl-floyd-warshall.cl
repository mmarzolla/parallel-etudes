/****************************************************************************
 *
 * opencl-floyd-warshall.cl - OpenCL kernels for the Floyd-Warshall algorithm.
 *
 * Copyright (C) 2025 Moreno Marzolla
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

typedef struct {
    int src, dst;
    float w;
} edge_t;

int IDX(int i, int j, int width)
{
    return i * width + j;
}

__kernel
void fw_init1( __global float *d, __global int *p, int n )
{
    const int u = get_global_id(1);
    const int v = get_global_id(0);

    if ((u < n) && (v < n)) {
        d[IDX(u,v,n)] = (u == v ? 0.0 : HUGE_VAL);
        p[IDX(u,v,n)] = -1;
    }
}

__kernel
void fw_init2(__global const edge_t *e,
              __global float *d,
              __global int *p,
              int n, int m)
{
    const int i = get_global_id(0);
    if (i < m) {
        d[IDX(e[i].src,e[i].dst,n)] = e[i].w;
        p[IDX(e[i].src,e[i].dst,n)] = e[i].src;
    }
}

void fw_relax(__global float *d,
              __global int *p,
              int u, int v, int k, int n)
{
    if (d[IDX(u,k,n)] + d[IDX(k,v,n)] < d[IDX(u,v,n)]) {
        d[IDX(u,v,n)] = d[IDX(u,k,n)] + d[IDX(k,v,n)];
        p[IDX(u,v,n)] = p[IDX(k,v,n)];
    }
}

/* Executed by one thread only; relax (k,k). */
__kernel
void fw_relax0(__global float *d,
               __global int *p,
               int k, int n)
{
    if (get_global_id(0) == 0)
        fw_relax(d, p, k, k, k, n);
}

/* Executed by n threads; relax (k, *) and (*, k). */
__kernel
void fw_relax1(__global float *d,
               __global int *p,
               int k, int n)
{
    const int i = get_global_id(0);
    if ((i < n) && (i != k)) {
        fw_relax(d, p, i, k, k, n);
        fw_relax(d, p, k, i, k, n);
    }
}

/* Executed by n x n threads; relax everything else. */
__kernel
void fw_relax2(__global float *d,
               __global int *p,
               int k, int n)
{
    const int v = get_global_id(1);
    const int u = get_global_id(0);
    if ((u < n) && (v < n) && (u != k) && (v != k))
        fw_relax(d, p, u, v, k, n);
}

__kernel
void fw_check(__global const float *d,
              int n,
              __global int *result)
{
    const int u = get_global_id(0);
    if ((u < n) && (d[IDX(u,u,n)] < 0.0)) {
        // no need to protect against race conditions here
        *result = 1;
    }
}
