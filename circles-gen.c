/****************************************************************************
 *
 * circles-gen.c - Generate an input file for the mpi-circles.c program
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * gcc -ansi -Wall -Wpedantic circles-gen.c -o circles-gen
 *
 * To generate 1000 random rectangles, run:
 * ./circles-gen 1000 > circles-1000.in
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
