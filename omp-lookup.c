/****************************************************************************
 *
 * omp-lookup.c - Parallel linear search
 *
 * Copyright (C) 2023 by Alice Girolomini <alice.girolomini(at)studio.unibo.it>
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

/***
% HPC - Parallel linear search
% Alice Girolomini <alice.girolomini@studio.unibo.it>
% Last updated: 2023-09-27

Write an OMP program that finds the positions of all occurrences of a
given `key` in an unsorted integer array `v[]`. For example, if `v[] =
{1, 3, -2, 3, 4, 3, 3, 5, -10}` and `key = 3`, the program must
build the result array

        {1, 3, 5, 6}
To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-lookup.c -o omp-lookup

To execute:

        ./omp-lookup [N]

Example:

        ./omp-lookup [N]

## Files

- [omp-lookup.c](omp-lookup.c)

***/

#include <omp.h>
#include <stdlib.h> /* for rand() */
#include <time.h>   /* for time() */
#include <stdio.h>
#include <assert.h>

void fill (int *v, int n) {
    int i;
    for (i = 0; i < n; i++) {
        v[i] = (rand() % 100);
    }
}

int main (int argc, char *argv[]) {

    int n = 1000;       /* Lenght of the input array */
    int *v = NULL;      /* Input array */
    int *result = NULL; /* Array which contains the indexes of the occurrences */
    int nf = 0;         /* Total occurrences */
    const int KEY = 42; /* The value to search for */
    int i;


    if (argc > 1){
        n = atoi(argv[1]);

        v = (int*)malloc(n * sizeof(*v)); 
        assert(v != NULL);
        fill(v, n);
    }

    const double tstart = omp_get_wtime();
#ifdef SERIAL
    /* Counts the number of occurrences of `KEY` in `v[]` */
    for (i = 0; i < n; i++) {
        if (v[i] == KEY){
            nf++;
        }
    }

    /* Allocates the result array */
    result = (int*) malloc(nf * sizeof(*result)); 
    assert(result != NULL);

    /* Fills the result array  */
    int r = 0;
    for (i = 0; i < n; i++) {
        if (v[i] == KEY) {
            result[r] = i;
            r++;
        }
    }
#else
    const int max_threads = omp_get_max_threads();
    int *my_nf = (int*)calloc(max_threads, max_threads * sizeof(int));
    
    /** 
     * Every thread counts the local number of occurrences of `KEY` in `v[]`, 
     * then stores the value in my_nf[id_thread] 
     * and eventually performs a reduction 
     */
    #if __GNUC__ < 9 
    #pragma omp parallel for default(none) shared(n, v, my_nf) reduction(+ : nf)
    #else
    #pragma omp parallel for default(none) shared(n, v, my_nf, KEY) reduction(+ : nf)
    #endif
    for (i = 0; i < n; i++) {
        if (v[i] == KEY){
            nf++;
            my_nf[omp_get_thread_num()] = nf;
        }
    }

    result = (int*)malloc(nf * sizeof(*result)); 
    assert(result != NULL);

    /** 
     * Every thread computes its portion of the result array and then fills it 
     * with the positions of the occurrences 
     */
    #if __GNUC__ < 9 
    #pragma omp parallel default(none) shared(n, result, v, nf, my_nf) private(i)
    #else
    #pragma omp parallel default(none) shared(n, result, v, nf, my_nf, KEY) private(i)
    #endif
    {
        const int id_thread = omp_get_thread_num();
        const int n_threads = omp_get_num_threads();
        int my_start = n * id_thread / n_threads;
        int my_end = n * (id_thread + 1) / n_threads;
        int my_r = 0;

        for(i = 0; i < id_thread; i++){
            my_r += my_nf[i];
        }

        int end_r = my_r + my_nf[id_thread];

        for (i = my_start; i < my_end; i++) {
            if (v[i] == KEY && my_r <  end_r){
                result[my_r] = i;
                my_r++;
            }
        }
    }

    free(my_nf);
#endif
    const double elapsed = omp_get_wtime() - tstart;
    fprintf(stderr,"\nExecution time %f seconds\n", elapsed);

    printf("There are %d occurrences of %d\n", nf, KEY);
    printf("Positions: ");
    for (int i = 0; i < nf; i++) {
        printf("%d ", result[i]);
        if (v[result[i]] != KEY) {
            fprintf(stderr, "\nFATAL: v[%d]=%d, expected %d\n", result[i], v[result[i]], KEY);
        }
    }
    printf("\n");
    free(v);
    free(result);

    return EXIT_SUCCESS;
}