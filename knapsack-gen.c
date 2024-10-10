/****************************************************************************
 *
 * knapsack-gen.c - generate instances for the knapsack solver
 *
 * Written in 2016, 2017 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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

#include <stdio.h>
#include <stdlib.h>
#include <time.h> /* for time() */

int main( int argc, char* argv[] )
{
    int C, n;
    if ( argc != 3 ) {
        fprintf(stderr, "Usage: %s knapsack_capacity num_items\n", argv[0]);
        return EXIT_FAILURE;
    }
    srand(time(NULL));
    C = atoi(argv[1]);
    n = atoi(argv[2]);
    printf("%d\n%d\n", C, n);
    for ( int i=0; i<n; i++ ) {
        printf("%d %f\n", 1 + rand() % (C/2), ((double)rand())/RAND_MAX * 10.0);
    }
    return EXIT_SUCCESS;
}
