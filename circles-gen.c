/****************************************************************************
 *
 * circles-gen.c - Generate an input file for the mpi-circles.c program
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 *
 *      gcc -ansi -Wall -Wpedantic circles-gen.c -o circles-gen
 *
 * To generate 1000 random rectangles, run:
 *
 *      ./circles-gen 1000 > circles-1000.in
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>

float randab(float a, float b)
{
    return a + (rand() / (float)RAND_MAX)*(b-a);
}

int main( int argc, char* argv[] )
{
    int i, n;
    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s ncircles\n", argv[0]);
        return EXIT_FAILURE;
    }
    n = atoi( argv[1] );
    printf("%d\n", n);
    for (i=0; i<n; i++) {
        const float r = randab(1, 20);
        const float x = randab(r, 1000-r);
        const float y = randab(r, 1000-r);
        printf("%f %f %f\n", x, y, r);
    }
    return EXIT_SUCCESS;
}
