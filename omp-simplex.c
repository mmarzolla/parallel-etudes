/****************************************************************************
 *
 * omp-simplex.c - Solve LP Problem with Primal Simplex Algorithm
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
% HPC - Parallel Primal Simplex Algorithm
% Alice Girolomini <alice.girolomini@studio.unibo.it>
% Last updated: 2023-08-24

Solves LP Problem with Primal Simplex: { minimize cx : Ax <= b, x >= 0 }.
Input: { m, n, Mat[m x n] } where
b = mat[1..m,0] .. column 0 is b >= 0
c = mat[0,1..n] .. row 0 is z to minimize, c is negated in input
A = mat[1..m,1..n] .. constraints
x = [x1..xm] are the variables
Slack variables are already in the input

Example input file for read_tableau:
    4 7
    0.000 2.000 1.000 -1.000 0.000 0.000 0.000
    4.000 1.000 1.000 2.000 1.000 0.000 0.000
    2.000 -1.000 2.000 1.000 0.000 1.000 0.000
    2.000 1.000 -1.000 -1.000 -0.000 -0.000 1.000 

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-simplex.c -o omp-simplex

To execute:

        ./omp-simplex input_file.txt

Example:

        ./omp-simplex matrix.txt

## Files

- [omp-simplex.c](omp-simplex.c)

***/
#include <omp.h>
#include <stdlib.h>
#include <time.h>  
#include <stdio.h>
#include <assert.h>
#include <string.h>

#define UNBOUNDED -1

typedef struct {
  int m, n;
  double **mat;
} Tableau;

typedef struct {
  int row, column;
  double value;
} Pivot;

/* Check whether b = mat[1..m,0] is >= 0 */
void check_b_positive(Tableau *tab) {

    for(int i = 1; i < tab->m; i++){
        if(tab->mat[i][0] < 0){
            fprintf(stderr, "\nFATAL: b[%d] must be positive\n", i);
            exit(1);
        }
    }
}

void print_tableau(Tableau *tab) {

  printf("\n Tableau:\n");
  for (int i = 0; i < tab->m; i++) {
    for (int j = 0; j < tab->n; j++) {
        printf(" %lf", tab->mat[i][j]);
    }
    printf("\n");
  }
}

void print_solution(Tableau *tab) {
    int i, j;
    int *x = (int*) calloc((tab->m - 1), sizeof(int));
    int *row = (int*) malloc((tab->m - 1) * sizeof(int));
  
    printf("Solutions: \n");
    printf("    Cost: %f\n", tab->mat[0][0]);
    for (i = 1; i < tab->m; i++){
        for (j = 1; j < tab->n; j++){
            if (tab->mat[i][j] == 1 && tab->mat[0][j] == 0 && j < tab->m){
                x[j-1] ++;
                row[j-1] = i;
            }
        }
    }

    for (i = 0; i < tab->m - 1; i++){
        if (x[i] == 1){
            printf("X%d = %lf\n", i+1, tab->mat[row[i]][0]);
        }
    }

} 

/* Read tableau from file */
void read_tableau (Tableau *tab, const char * filename) {
    int err, i, j;
    FILE *fp;

    fp  = fopen(filename, "r");
    if (!fp) {
        printf("Cannot read %s\n", filename); 
        exit(1);
    }

    err = fscanf(fp, "%d %d", &tab->m, &tab->n);

    if (err < 2 || err == EOF) {
        printf("Cannot read m or n\n"); 
        exit(1);
    }

    tab->mat = (double**) malloc(tab->m * sizeof(double*));
    for (i = 0; i < tab->m; i++){
        tab->mat[i] = (double*) malloc(tab->n * sizeof(double));
    }

    for (i = 0; i < tab->m; i++) {
        for (j = 0; j < tab->n; j++) {
            err = fscanf(fp, "%lf", &tab->mat[i][j]);
            if (err == 0 || err == EOF) {
                printf("Cannot read A[%d][%d]\n", i, j); 
                exit(1);
            } 
        }   
    }

    check_b_positive(tab);

    printf("Read tableau [%d rows x %d columns] from file '%s'.\n", tab->m, tab->n, filename);
    fclose(fp);

}

#ifdef SERIAL

/* Select pivot column */
/*  Select the greatest value in mat[0][1..n] */
int find_pivot_col (Tableau *tab){
    int pivot_col = 1;
    double highest_val = 0;

    for (int j = 1; j < tab->n; j++) {
        if (tab->mat[0][j] > highest_val && tab->mat[0][j] != 0) {
            highest_val = tab->mat[0][j];
            pivot_col = j;
        }
    }

    if(highest_val == 0){
        return 0;
    }

    return pivot_col;
}

/* Select pivot row */
/* Count the number of positive values in the given column, if all are < 0 then solution is unbounded else finds the smallest positive ratio min_ratio = mat[0] / mat[pivot_col] */
int find_pivot_row (Tableau *tab, int pivot_col){
    int pivot_row = 0;
    double min_ratio = -1;

    for (int i = 1; i < tab->m; i++) {
        if (tab-> mat[i][pivot_col] > 0.0) {
            double ratio = tab->mat[i][0] / tab->mat[i][pivot_col];
            if ((ratio > 0 && ratio < min_ratio) || min_ratio < 0) {
                min_ratio = ratio;
                pivot_row = i;
            }
        }
    }

    if (min_ratio == UNBOUNDED){
        fprintf(stderr, "Unbounded solution\n");
        exit(1);
    }
    printf("    Pivot row %d\n", pivot_row);

    return pivot_row;
}

/* Update pivot row */
/* Convert pivot element to 1 and updates the other element in the row */
void update_pivot_row (Tableau *tab, int pivot_row, double pivot){

    for (int j = 0; j < tab->n; j++) {
        tab->mat[pivot_row][j] = tab->mat[pivot_row][j] / pivot;
    }
    
}

/* Update rows */
/* Updates all other rows except the pivot row*/
void update_rows (Tableau *tab, int pivot_row, int pivot_col){
    double coeff;

    for (int i = 0; i < tab->m; i++) {
        if (i != pivot_row) {
            coeff = -tab->mat[i][pivot_col];
            for (int j = 0; j < tab->n; j++) {
                tab->mat[i][j] = (coeff * tab->mat[pivot_row][j]) + tab->mat[i][j];
            }
        }
    }

   /*print_tableau(tab);*/
}

#else

/* Select pivot column */
int find_pivot_col (Tableau *tab){
    int j, pivot_col;
    const int max_threads = omp_get_max_threads();
    int *local_max_index = (int*)malloc(max_threads * sizeof(int));
    double *local_max = (double*)calloc(max_threads, sizeof(double));
    double highest_val = 0;

    #pragma omp parallel default(none) shared(tab, local_max_index, local_max) private(j)
    {
        const int id_thread = omp_get_thread_num();
        const int n_threads = omp_get_num_threads();
        int my_start = tab->n * id_thread / n_threads;
        int my_end = tab->n * (id_thread + 1) / n_threads;

        for (j = my_start; j < my_end; j++) {
            if (tab->mat[0][j] > local_max[id_thread] && tab->mat[0][j] != 0 && j != 0) {
                local_max[id_thread] = tab->mat[0][j];
                local_max_index[id_thread]= j;
            }
        }
    }

    for (j = 0; j < max_threads; j++) {
        if (local_max[j] > highest_val) {
            highest_val = local_max[j];
            pivot_col = local_max_index[j];
        }
    }

    printf("%lf %d\n", highest_val, pivot_col);
    if(highest_val == 0){
        return 0;
    }

    return pivot_col;
}

/* Select pivot row */
int find_pivot_row (Tableau *tab, int pivot_col){
    int i, pivot_row = 0;
    const int max_threads = omp_get_max_threads();
    int *local_min_index = (int*)malloc(max_threads * sizeof(int));
    double *local_min = (double*)malloc(max_threads * sizeof(double));
    double min_ratio = -1;

    #pragma omp parallel default(none) shared(tab, local_min_index, local_min, pivot_col) private(i)
    {
        const int id_thread = omp_get_thread_num();
        const int n_threads = omp_get_num_threads();
        int my_start = tab->m * id_thread / n_threads;
        int my_end = tab->m * (id_thread + 1) / n_threads;
        local_min[id_thread] = -1;

        for (i = my_start; i < my_end; i++) {
            if (tab-> mat[i][pivot_col] > 0.0 && i != 0) {
                double ratio = tab->mat[i][0] / tab->mat[i][pivot_col];
                if ((ratio > 0 && ratio < local_min[id_thread]) || local_min[id_thread] < 0) {
                    local_min[id_thread] = ratio;
                    local_min_index[id_thread] = i;
                }
            }
        }
    }

    for (i = 0; i < max_threads; i++) {
        if ((local_min[i] > 0 && local_min[i] < min_ratio) || min_ratio < 0) {
            min_ratio = local_min[i];
            pivot_row = local_min_index[i];
        }
    }

    if (min_ratio == UNBOUNDED){
        fprintf(stderr, "Unbounded solution\n");
        exit(1);
    }
    printf("    Pivot row %d\n", pivot_row);

    return pivot_row;
}

/* Update pivot row */
void update_pivot_row (Tableau *tab, int pivot_row, double pivot){

    #pragma omp parallel for default(none) shared(tab,  pivot, pivot_row)
    for (int j = 0; j < tab->n; j++) {
        tab->mat[pivot_row][j] = tab->mat[pivot_row][j] / pivot;
    }
    
}

/* Update rows */
void update_rows (Tableau *tab, int pivot_row, int pivot_col){
    int i, j;
    double *coeff = (double*)malloc(tab->m * sizeof(double));

    #pragma omp parallel for default(none) shared(tab, pivot_col, coeff)
    for (i = 0; i < tab->m; i++) {
        coeff[i] = -tab->mat[i][pivot_col];
    }

    #pragma omp parallel for collapse(2) default(none) shared(tab, pivot_row, pivot_col, coeff)
    for (i = 0; i < pivot_row; i++) {
        for (j = 0; j < tab->n; j++) {
            tab->mat[i][j] = (coeff[i] * tab->mat[pivot_row][j]) + tab->mat[i][j];
        }
    }

    #pragma omp parallel for collapse(2) default(none) shared(tab, pivot_row, pivot_col, coeff)
    for (i = pivot_row + 1; i < tab->m; i++) {
        for (j = 0; j < tab->n; j++) {
            tab->mat[i][j] = (coeff[i] * tab->mat[pivot_row][j]) + tab->mat[i][j];
        }
    }

    /*print_tableau(tab);*/
}
#endif

int main ( int argc, char *argv[] ) {

    int it = 0, optimal = 0;
    Pivot p;
    Tableau tab;
    
    if (argc != 2) {
        fprintf(stderr, "Missing matrix\n");
        return EXIT_FAILURE;
    }

    read_tableau(&tab, argv[1]);
    const double tstart = omp_get_wtime();
    do {
        p.column = find_pivot_col(&tab);

        if (p.column == 0){
            optimal = 1;
        } else {
            it++;
            printf("Iteration: %d\n", it);
            printf("    Pivot column %d\n", p.column);

            p.row = find_pivot_row(&tab, p.column);
            p.value = tab.mat[p.row][p.column];
            update_pivot_row(&tab, p.row, p.value);
            update_rows(&tab, p.row, p.column);

        }
    } while (optimal == 0); 

    const double elapsed = omp_get_wtime() - tstart;

    print_solution(&tab);
    printf("Number of iterations: %d\n", it);
    fprintf(stderr,"\nExecution time %f seconds\n", elapsed);

    return 0;

}