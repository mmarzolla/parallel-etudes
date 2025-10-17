/****************************************************************************
 *
 * gen-bbox.c - Generate an input file for the mpi-bbox.c program
 *
 * Copyright (C) 2017, 2022 Moreno Marzolla
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
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -ansi -Wall -Wpedantic gen-bbox.c -o gen-bbox
 *
 * To generate 1000 random rectangles, run:
 * ./gen-bbox 1000 > bbox-1000.in
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>

/* Generate a random number in [a, b] */
float randab(float a, float b)
{
    return a + (rand() / (float)RAND_MAX)*(b-a);
}

/* If necessary, exchange *x and *y so that at the end we have *x <=
   *y */
void compare_and_swap( float *x, float *y )
{
    if (*x > *y) {
        float tmp = *x;
        *x = *y;
        *y = tmp;
    }
}

int main( int argc, char* argv[] )
{
    int n;
    if ( argc != 2 ) {
        printf("Usage: %s n\n", argv[0]);
        return EXIT_FAILURE;
    }
    n = atoi( argv[1] );
    printf("%d\n", n);
    for (int i=0; i<n; i++) {
        float x1 = randab(0, 1000), x2 = randab(0, 1000);
        float y1 = randab(0, 1000), y2 = randab(0, 1000);
        compare_and_swap(&x1, &x2);
        compare_and_swap(&y1, &y2);
        printf("%f %f %f %f\n", x1, y2, x2, y1);
    }
    return EXIT_SUCCESS;
}
