/****************************************************************************
 *
 * bbox-gen.c - Generate an input file for the mpi-bbox.c program
 *
 * Copyright (C) 2017, 2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -ansi -Wall -Wpedantic bbox-gen.c -o bbox-gen
 *
 * To generate 1000 random rectangles, run:
 * ./bbox-gen 1000 > bbox-1000.in
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
    int i, n;
    if ( argc != 2 ) {
        printf("Usage: %s n\n", argv[0]);
        return EXIT_FAILURE;
    }
    n = atoi( argv[1] );
    printf("%d\n", n);
    for (i=0; i<n; i++) {
        float x1 = randab(0, 1000), x2 = randab(0, 1000);
        float y1 = randab(0, 1000), y2 = randab(0, 1000);
        compare_and_swap(&x1, &x2);
        compare_and_swap(&y1, &y2);
        printf("%f %f %f %f\n", x1, y2, x2, y1);
    }
    return EXIT_SUCCESS;
}
