/****************************************************************************
 *
 * mpi-simplex.c - Solve LP Problem with Primal Simplex Algorithm
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
% Last updated: 2023-10-31

Solves LP Problem with Primal Simplex: z_p = Min cx s.t. Ax >= b, x >= 0
Input: { m, n, Mat[m \times n] } where
b = mat[1..m,0] .. column 0 is b >= 0
c = mat[0,1..n] .. row 0 is Z to minimize, c is negated in input
A = mat[1..m,1..n] .. constraints
x = [x_1..x_m] are the variables
Slack variables are already in the input

Example input file for read_tableau:
    4 7
    0.000 2.000 1.000 -1.000 0.000 0.000 0.000
    4.000 1.000 1.000 2.000 1.000 0.000 0.000
    2.000 -1.000 2.000 1.000 0.000 1.000 0.000
    2.000 1.000 -1.000 -1.000 -0.000 -0.000 1.000 

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-simplex.c -o mpi-simplex

To execute:

        mpirun -n P ./mpi-simplex input_file.txt

Example:

        mpirun -n 4 ./mpi-simplex matrix.txt

## Files

- [mpi-simplex.c](mpi-simplex.c)

***/
#include <mpi.h>
#include <stdlib.h>
#include <time.h>  
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "hpc.h"

#define UNBOUNDED -1

typedef struct {
  int m, n;
  double *mat;
} Tableau;

typedef struct {
  int row, column;
  double value;
} Pivot;

/* Check whether b = mat[1..m,0] is >= 0 */
void check_b_positive (Tableau *tab) {

    for(int i = 1; i < tab->m; i++){
        if(tab->mat[i * tab->n] < 0){
            fprintf(stderr, "\nFATAL: b[%d] must be positive\n", i);
            exit(1);
        }
    }
}

void print_tableau(Tableau *tab) {

  printf("\n Tableau:\n");
  for (int i = 0; i < tab->m; i++) {
    for (int j = 0; j < tab->n; j++) {
        printf(" %lf", tab->mat[i * tab->n + j]);
    }
    printf("\n");
  }
}

void print_solution (Tableau *tab) {
    int *x = (int*) calloc((tab->m - 1), sizeof(int));
    int *row = (int*) malloc((tab->m - 1) * sizeof(int));
  
    printf("Solutions: \n");
    printf("    Cost: %f\n", tab->mat[0]);
    for (int i = 1; i < tab->m; i++) {
        for (int j = 1; j < tab->n; j++) {
            if (tab->mat[i * tab->n + j] == 1 && tab->mat[j] == 0 && j < tab->m) {
                x[j-1] ++;
                row[j-1] = i;
            }
        }
    }

    for (int i = 0; i < tab->m - 1; i++) {
        if (x[i] == 1){
            printf("X%d = %lf\n", i+1, tab->mat[row[i] * tab->n]);
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

    tab->mat = (double*) malloc(tab->m * tab->n * sizeof(double));

    for (i = 0; i < tab->m; i++) {
        for (j = 0; j < tab->n; j++) {
            err = fscanf(fp, "%lf", &tab->mat[i * tab->n + j]);
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

/** 
 * Updates all other rows except the pivot row
*/
void update_rows (Tableau *tab, int pivot_row, int pivot_col) {
    double coeff;

    for (int i = 0; i < tab->m; i++) {
        if (i != pivot_row) {
            coeff = -tab->mat[i * tab->n + pivot_col];
            for (int j = 0; j < tab->n; j++) {
                tab->mat[i * tab->n + j] = (coeff * tab->mat[pivot_row * tab->n + j]) + tab->mat[i * tab->n + j];
            }
        }
    }

   /*print_tableau(tab);*/
}

#ifdef SERIAL

/**
 * Selects the greatest value in mat[0][1..n] 
 * which represents the index  of the 
 * pivot column
 */
int find_pivot_col (Tableau *tab) {
    int pivot_col = 1;
    double highest_val = 0;

    for (int j = 1; j < tab->n; j++) {
        if (tab->mat[j] > highest_val && tab->mat[j] != 0) {
            highest_val = tab->mat[j];
            pivot_col = j;
        }
    }

    if(highest_val == 0) {
        return 0;
    }

    return pivot_col;
}

/** 
 * Checks the number of positive values in the pivot column, 
 * if all are < 0 then the solution is unbounded, else finds the 
 * smallest positive ratio min_ratio = mat[0] / mat[pivot_col]
 * which represents the pivot row 
*/
int find_pivot_row (Tableau *tab, int pivot_col) {
    int pivot_row = 0;
    double min_ratio = -1;

    for (int i = 1; i < tab->m; i++) {
        if (tab-> mat[i * tab->n + pivot_col] > 0.0) {
            double ratio = tab->mat[i * tab->n] / tab->mat[i * tab->n + pivot_col];
            if ((ratio > 0 && ratio < min_ratio) || min_ratio < 0) {
                min_ratio = ratio;
                pivot_row = i;
            }
        }
    }

    if (min_ratio == UNBOUNDED) {
        return min_ratio;
    }
    printf("    Pivot row %d\n", pivot_row);

    return pivot_row;
}

/** 
 * Converts pivot value to 1 and updates other elements in the row 
*/
void update_pivot_row (Tableau *tab, int pivot_row, double pivot) {

    for (int j = 0; j < tab->n; j++) {
        tab->mat[pivot_row * tab->n + j] = tab->mat[pivot_row * tab->n + j] / pivot;
    }
    
}

#else
/**
 * Selects the greatest value in mat[0][1..n] which represents 
 * the index of the pivot column
 */
int find_pivot_col (double *tab, int col, int n) {
    int my_rank, comm_sz, pivot_col = 1, j = 0;
    double highest_val = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        j++;
    }
    int start = n * my_rank /comm_sz;

    for (; j < col; j++) {
        if (tab[j] > highest_val && tab[j] != 0) {
            highest_val = tab[j];
            pivot_col = j + start;
        }
    }

    if (highest_val == 0) {
        return 0;
    }

    return pivot_col;
}

/** 
 * Checks the number of positive values in the pivot column, 
 * if all are < 0 then the solution is unbounded else finds the 
 * smallest positive ratio min_ratio = mat[0] / mat[pivot_col]
 * which represents the pivot row 
*/
int find_pivot_row (double *tab, int pivot_col, int rows, int m, int n) {
    int my_rank, comm_sz, pivot_row = 0;
    double min_ratio = -1;

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int start = m * my_rank /comm_sz;

    for (int i = 0; i < rows; i++) {
        if (tab[i * n + pivot_col] > 0.0) {
            double ratio = tab[i * n] / tab[i * n + pivot_col];
            if ((ratio > 0 && ratio < min_ratio) || min_ratio < 0) {
                min_ratio = ratio;
                pivot_row = start + i + 1;
            }
        }
    }

    return pivot_row;
}

/** 
 * Converts pivot value to 1 and updates other elements in the row 
*/
void update_pivot_row (double *tab, double pivot, int n) {

    for (int j = 0; j < n; j++) {
        tab[j] = tab[j] / pivot;
    }
    
}
#endif

int main (int argc, char *argv[]) {

    int my_rank, comm_sz;
    int it = 0, optimal = 0;
    Tableau tab;
    Pivot p;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if (argc != 2) {
        fprintf(stderr, "Missing matrix\n");
        return EXIT_FAILURE;
    }

    if (my_rank == 0) {
        read_tableau(&tab, argv[1]);
    }
    
    const double tstart = hpc_gettime();
#ifdef SERIAL
    if (my_rank == 0) {
        do {
            p.column = find_pivot_col(&tab);
            if (p.column == 0){
                optimal = 1;
            } else {
                /*print_tableau(&tab);*/
                it++;
                printf("Iteration: %d\n", it);
                printf("    Pivot column %d\n", p.column);
                p.row = find_pivot_row(&tab, p.column);
                p.value = tab.mat[p.row * tab.n + p.column];
                update_pivot_row(&tab, p.row, p.value);
                update_rows(&tab, p.row, p.column);
            }
        } while (optimal == 0); 
    }
#else
    int count, m;
    if (my_rank == 0) {
        count = tab.n;
        m = tab.m;
    }
    /* Broadcasts matrix dimensions to other processes */
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Defines partial result arrays */
    int max_col_index[comm_sz], min_ratio_index[comm_sz];
    /* Defines rowtype datatype */
    MPI_Datatype rowtype; 
    MPI_Type_contiguous(count, MPI_DOUBLE, &rowtype);
    MPI_Type_commit(&rowtype);

    /* Calculates sendcnt and displacements for each process */
    int *row_sendcounts = (int*) malloc(comm_sz * sizeof(*row_sendcounts));
    int *col_sendcounts = (int*) malloc(comm_sz * sizeof(*col_sendcounts));
    int *row_displs = (int*) malloc(comm_sz * sizeof(*row_displs));
    int *col_displs = (int*) malloc(comm_sz * sizeof(*col_displs));
    for (int i = 0; i < comm_sz; i++) {
        const int row_start = (m - 1) * i / comm_sz;
        const int row_end = (m - 1) * (i + 1) / comm_sz;
        const int col_start = count * i / comm_sz;
        const int col_end = count * (i + 1) / comm_sz;
        row_sendcounts[i] = row_end - row_start;
        row_displs[i] = row_start;
        col_sendcounts[i] = col_end - col_start;
        col_displs[i] = col_start;
    }
    
    double *local_mat = (double*) malloc(row_sendcounts[my_rank] * count * sizeof(double));
    assert(local_mat != NULL);
    double *local_v = (double*) malloc(col_sendcounts[my_rank] * sizeof(double));
    assert(local_v != NULL);

    do {
        /* Scatters the cost coefficients array then each process finds the local maximum value */
        MPI_Scatterv(&tab.mat[0], col_sendcounts, col_displs, MPI_DOUBLE, &local_v[0], col_sendcounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        max_col_index[my_rank] = find_pivot_col(local_v, col_sendcounts[my_rank], count);
        MPI_Gather(&max_col_index[my_rank], 1, MPI_INT, max_col_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
        /* Process 0 finds the global maximum value which is the pivot column */
        if (my_rank == 0) {
            double highest_val = 0; 
            for (int i = 0; i < comm_sz; i++) {
                if (tab.mat[max_col_index[i]] > highest_val && max_col_index[i] != 0) {
                    p.column = max_col_index[i];
                    highest_val = tab.mat[max_col_index[i]];
                }
            }
            if (highest_val == 0) {
                p.column = 0;
            }
        }

        MPI_Bcast(&p.column, 1, MPI_INT, 0, MPI_COMM_WORLD);
        /* Checks whether the pivot column has been found */
        if (p.column == 0) {
            optimal = 1;
        } else {
            if (my_rank == 0) {
                /*print_tableau(&tab);*/
                it++;
                printf("Iteration: %d\n", it);
                printf("    Pivot column %d\n", p.column);
            }

            /* Scatters the matrix in order to find the minimum ratio br/yr = min{bi/yi: yi > 0, i = 1, . . . ,m} */
            MPI_Scatterv(&tab.mat[count], row_sendcounts, row_displs, rowtype, &local_mat[0], row_sendcounts[my_rank], rowtype, 0, MPI_COMM_WORLD);
            min_ratio_index[my_rank] = find_pivot_row(local_mat, p.column, row_sendcounts[my_rank], m-1, count);
            MPI_Gather(&min_ratio_index[my_rank], 1, MPI_INT, min_ratio_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
            /* Process 0 finds the global minimum ratio */
            if (my_rank == 0) {
                double min_ratio = -1;
                for (int i = 0; i < comm_sz; i++) {
                    if(min_ratio_index[i] != 0) {
                        double ratio = tab.mat[min_ratio_index[i] * count] / tab.mat[min_ratio_index[i] * count + p.column];
                        if ((ratio > 0 && ratio < min_ratio) || min_ratio < 0) {
                            min_ratio = ratio;
                            p.row = min_ratio_index[i];
                        }
                    }
                }
                /* If the min ratio < 0 the problem is unbounded */
                if (min_ratio == UNBOUNDED) {
                    fprintf(stderr, "Unbounded solution\n");
                    return EXIT_FAILURE;
                }
            }

            MPI_Bcast(&p.row, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (my_rank == 0) {  
                printf("    Pivot row %d\n", p.row); 
                p.value = tab.mat[p.row * count + p.column];
            }
            /* Scatters the pivot value */
            MPI_Bcast(&p.value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            /* Scatters the pivot row and each process updates its part of the row*/
            MPI_Scatterv(&tab.mat[p.row * count], col_sendcounts, col_displs, MPI_DOUBLE, &local_v[0], col_sendcounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
            update_pivot_row(local_v, p.value, col_sendcounts[my_rank]);
            MPI_Gatherv(&local_v[0], col_sendcounts[my_rank], MPI_DOUBLE, &tab.mat[p.row * count], col_sendcounts, col_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            /* Process 0 updates the rest of the matrix */
            if (my_rank == 0) {
                update_rows(&tab, p.row, p.column);
            }
        }
    } while (optimal == 0); 

    free(local_mat);
    free(local_v);
    free(rowtype);
#endif
    const double elapsed = hpc_gettime() - tstart;
    if(my_rank == 0) {
        print_solution(&tab);
        printf("Number of iterations: %d\n", it);
        fprintf(stderr,"\nExecution time %f seconds\n", elapsed);
        
        free(tab.mat);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;

}