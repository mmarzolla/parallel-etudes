/****************************************************************************
 *
 * opencl-merge-sort.cl - Kernels for opencl-merge-sort.c
 *
 * Copyright (C) 2026 Moreno Marzolla
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

/**
 * Merge two adjacent sub-arrays `src[low..mid]` and
 * `src[mid+1..high]`, put the result in `dst[low..high]`.
 */
__kernel void
merge_kernel(const __global int* src,
	     int len,
	     int n,
	     __global int* dst)
{
    /* avoid overflow of `low` below. */
    if (get_global_id(0) >= (n + 2*len-1) / (2*len))
        return;

    const int low = get_global_id(0)*2*len;
    const int mid = min(n-1, low+len-1);
    const int high = min(n-1, mid+len);

    int i=low, j=mid+1, k=low;
    while (i<=mid && j<=high) {
        if (src[i] <= src[j]) {
            dst[k] = src[i++];
        } else {
            dst[k] = src[j++];
        }
        k++;
    }
    /* Handle leftovers */
    while (i<=mid) {
        dst[k] = src[i++];
        k++;
    }
    while (j<=high) {
        dst[k] = src[j++];
        k++;
    }
}
