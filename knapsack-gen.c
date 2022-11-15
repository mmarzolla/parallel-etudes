/****************************************************************************
 *
 * knapsack-gen.c - generate instances for the knapsack solver
 *
 * Written in 2016, 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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

#include <stdio.h>
#include <stdlib.h>
#include <time.h> /* for time() */

int main( int argc, char* argv[] )
{
    int i, C, n;
    if ( argc != 3 ) {
        fprintf(stderr, "Usage: %s knapsack_capacity num_items\n", argv[0]);
        return EXIT_FAILURE;
    }
    srand(time(NULL));
    C = atoi(argv[1]);
    n = atoi(argv[2]);
    printf("%d\n%d\n", C, n);
    for ( i=0; i<n; i++ ) {
        printf("%d %f\n", 1 + rand() % (C/2), ((float)rand())/RAND_MAX * 10.0);
    }
    return EXIT_SUCCESS;
}
