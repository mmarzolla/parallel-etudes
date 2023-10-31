/****************************************************************************
 *
 * cuda-simplex.cu - Solve LP Problem with Primal Simplex Algorithm
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

        nvcc cuda-simplex.cu -o cuda-simplex -lm

To execute:

        ./cuda-simplex input_file.txt

Example:

        ./cuda-simplex matrix.txt

## Files

- [cuda-simplex.cu](cuda-simplex.cu)

***/
#include <stdlib.h>
#include <time.h>  
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "hpc.h"

#define BLKDIM 1024
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

    for (int i = 1; i < tab->m; i++) {
        if (tab->mat[i * tab->n] < 0) {
            fprintf(stderr, "\nFATAL: b[%d] must be positive\n", i);
            exit(1);
        }
    }
}

void print_tableau (Tableau *tab) {

  printf("\n Tableau:\n");
  for (int i = 0; i < tab->m; i++) {
    for (int j = 0; j < tab->n; j++) {
        printf(" %lf", tab->mat[i * tab->n + j]);
    }
    printf("\n");
  }
}

void print_solution (Tableau *tab) {
    int i, j;
    int *x = (int*) calloc((tab->m - 1), sizeof(int));
    int *row = (int*) malloc((tab->m - 1) * sizeof(int));
  
    printf("Solutions: \n");
    printf("    Cost: %f\n", tab->mat[0]);
    for (i = 1; i < tab->m; i++) {
        for (j = 1; j < tab->n; j++) {
            if (tab->mat[i * tab->n + j] == 1 && tab->mat[j] == 0 && j < tab->m) {
                x[j-1] ++;
                row[j-1] = i;
            }
        }
    }

    for (i = 0; i < tab->m - 1; i++) {
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

    tab->mat = (double*) malloc(tab->n * tab->m * sizeof(double));

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
        if (tab->mat[j] > highest_val && tab->mat[j] !=0)  {
            highest_val = tab->mat[j];
            pivot_col = j;
        }
    }

    if(highest_val == 0){
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
        fprintf(stderr, "Unbounded solution\n");
        exit(1);
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

    print_tableau(tab);
}

#else

/**
 * Selects the greatest value in mat[0][1..n] 
 * which represents the index  of the 
 * pivot column
 */
__global__ void find_pivot_col (double *mat, int *pivot_col, double *highest_val, int n) {
    __shared__ double temp[BLKDIM];
    __shared__ int indexes[BLKDIM];
    const int tid = threadIdx.x;
    const int j = tid + blockIdx.x * blockDim.x;

    /* Every thread stores mat[0][j] */
    if (j > 0 && j < n) {
        temp[tid] = mat[j];
        indexes[tid] = j;
    } else {
        temp[tid] = 0;
    }
    __syncthreads(); 

    /* All threads within the block cooperate to find the maximum value in mat[0][1..n] */
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            int local_max_index = temp[tid] > temp[tid + offset] ? tid : tid + offset;
            indexes[tid] = indexes[local_max_index];
            temp[tid] = temp[local_max_index];
        }
        __syncthreads(); 
    }
    
    /* The thread of each block with tid 0 stores the local maximum */
    if (tid == 0) {
        highest_val[blockIdx.x] = temp[0];
        pivot_col[blockIdx.x] = indexes[0];
    }

}

/** 
 * Checks the number of positive values in the pivot column, 
 * if all are < 0 then the solution is unbounded, else finds the 
 * smallest positive ratio min_ratio = mat[0] / mat[pivot_col]
 * which represents the pivot row 
*/
__global__ void find_pivot_row (double *mat, int pivot_col, int *pivot_row, double *min_ratio, int n, int m) {
    __shared__ double temp[BLKDIM];
    __shared__ int indexes[BLKDIM];
    const int tid = threadIdx.x;
    const int i = tid + blockIdx.x * blockDim.x;
    
    /* Every thread stores the local ratio */
    if (i > 0 && i < m && mat[i * n + pivot_col] > 0) {
        temp[tid] = mat[i * n] / mat[i * n + pivot_col];
        indexes[tid] = i;
    } else {
        temp[tid] = 0;
    }
    __syncthreads(); 

    /* All threads within the block cooperate to find the minimun ratio */
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            if ((temp[tid + offset] < temp[tid] && temp[tid + offset] > 0) || (temp[tid] == 0 && temp[tid + offset] > 0)) {
                indexes[tid] = indexes[tid + offset];
                temp[tid] = temp[tid + offset];
            }
        }
        __syncthreads(); 
    }

    /* The thread of each block with tid 0 stores the local minimum */
    if (tid == 0) {
        min_ratio[blockIdx.x] = temp[0];
        pivot_row[blockIdx.x] = indexes[0];
    }

}

/** 
 * Converts pivot value to 1 and updates other elements in the row
*/
__global__ void update_pivot_row (double *mat, int pivot_col, int pivot_row, int n) {
    const int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < n) {
        double pivot = mat[pivot_row * n + pivot_col];
        mat[pivot_row * n + j] = mat[pivot_row * n + j] / pivot;
    }
    
}

/** 
 * Updates all other rows except the pivot row
*/
__global__ void update_rows (double *mat, int pivot_col, int pivot_row, int n, int i) {
    const int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < n) {
        double coeff = - mat[i * n + pivot_col];
        mat[i * n + j] = (coeff * mat[pivot_row * n + j]) + mat[i * n + j];
    }
    
}


/* Finds the maximum value and its index in the given array */
int find_max (double *local_result, int *indexes, int n_blocks) {
    double highest_val = 0;
    int pivot_col = 0;

    for (int i = 0; i < n_blocks; i++) {
        if (local_result[i] > highest_val) {
            highest_val = local_result[i] ;
            pivot_col = indexes[i];
        }
    }

    return pivot_col;
}

/* Finds the minimum value and its index in the given array*/
int find_min_ratio (double *local_result, int *indexes, int n_blocks) {
    double min_ratio = -1;
    int pivot_row = 0;
    
    for (int i = 0; i < n_blocks; i++) {
        if (local_result[i] < min_ratio || min_ratio < 0) {
            min_ratio = local_result[i] ;
            pivot_row = indexes[i];
        }
    }

    if (min_ratio < 0) {
        fprintf(stderr, "Unbounded solution\n");
        exit(1);
    }

    return pivot_row;
}

#endif

int main (int argc, char *argv[]) {

    int it = 0, optimal = 0;
    Pivot p;
    Tableau tab;
    
    if (argc != 2) {
        fprintf(stderr, "Missing matrix\n");
        return EXIT_FAILURE;
    }

    if (BLKDIM & (BLKDIM-1) != 0) {
        fprintf(stderr, "BLKDIM must be a power of two\n");
        return EXIT_FAILURE;
    } 

    read_tableau(&tab, argv[1]);
    const double tstart = hpc_gettime();
#ifdef SERIAL
    do {
        p.column = find_pivot_col(&tab);

        if (p.column == 0){
            optimal = 1;
        } else {
            it++;
            printf("Iteration: %d\n", it);
            printf("    Pivot column %d\n", p.column);

            p.row = find_pivot_row(&tab, p.column);
            p.value = tab.mat[p.row * tab.n + p.column];
            update_pivot_row(&tab, p.row, p.value);
            update_rows(&tab, p.row, p.column);

        }
    } while (optimal == 0); 
#else
    int *col_indexes, *row_indexes;
    int *d_col_indexes, *d_row_indexes;
    double *max_local_result, *min_local_result;
    double *d_mat, *d_max_local_result, *d_min_local_result;

    const size_t size = tab.n * tab.m * sizeof(*tab.mat);
    const int blocks_col = (tab.n + BLKDIM - 1) / BLKDIM;
    const int blocks_row = (tab.m + BLKDIM - 1) / BLKDIM;

    col_indexes = (int*) malloc(blocks_col * sizeof(int));
    assert(col_indexes != NULL);
    max_local_result = (double*) malloc(blocks_col * sizeof(double));
    assert(max_local_result != NULL);
    row_indexes = (int*) malloc(blocks_row * sizeof(int));
    assert(row_indexes != NULL);
    min_local_result = (double*) malloc(blocks_row * sizeof(double));
    assert(min_local_result != NULL);

    /* Allocates space for device copies */   
    cudaMalloc((void **)&d_mat, size);
    cudaMalloc((void **)&d_col_indexes, blocks_col * sizeof(*d_col_indexes));
    cudaMalloc((void **)&d_max_local_result, blocks_col * sizeof(*d_max_local_result));
    cudaMalloc((void **)&d_row_indexes, blocks_row * sizeof(*d_row_indexes));
    cudaMalloc((void **)&d_min_local_result, blocks_row * sizeof(*d_min_local_result));

    cudaMemcpy(d_mat, tab.mat, size, cudaMemcpyHostToDevice);
    do {
        /* Each independent block finds the max value in the cost coefficients array */
        find_pivot_col <<<blocks_col, BLKDIM>>> (d_mat, d_col_indexes, d_max_local_result, tab.n); 
        /* Copies the partial result from device to host */
        cudaMemcpy(max_local_result, d_max_local_result, blocks_col * sizeof(*max_local_result), cudaMemcpyDeviceToHost);
        cudaMemcpy(col_indexes, d_col_indexes, blocks_col * sizeof(*col_indexes), cudaMemcpyDeviceToHost);
        /* Host calculates the maximum value */
        p.column = find_max(max_local_result, col_indexes, blocks_col);
        if (p.column == 0) {
            optimal = 1;
        } else {
            it++;
            printf("Iteration: %d\n", it);
            printf("    Pivot column %d\n", p.column);

            /* Each independent block finds the minimum ratio in the pivot column */
            find_pivot_row <<<blocks_row, BLKDIM>>> (d_mat, p.column, d_row_indexes, d_min_local_result, tab.n, tab.m);
            /* Copies the partial result from device to host */
            cudaMemcpy(min_local_result, d_min_local_result, blocks_row * sizeof(*min_local_result), cudaMemcpyDeviceToHost);
            cudaMemcpy(row_indexes, d_row_indexes, blocks_row * sizeof(*row_indexes), cudaMemcpyDeviceToHost);
            /* Host calculates the minimum value */
            p.row = find_min_ratio(min_local_result, row_indexes, blocks_row);
            printf("    Pivot row %d\n", p.row);
            /* Updates pivot row */
            update_pivot_row <<<blocks_col, BLKDIM>>> (d_mat, p.column, p.row, tab.n);
            for (int i = 0; i< tab.m; i++) {
                if (i != p.row) {
                    /* Updates other rows */
                    update_rows <<<blocks_col, BLKDIM>>> (d_mat, p.column, p.row, tab.n, i);
                }
            }
        }
    } while (optimal == 0); 

    cudaMemcpy(tab.mat, d_mat, size, cudaMemcpyDeviceToHost);

    cudaFree(d_mat); 
    cudaFree(d_max_local_result);
    cudaFree(d_col_indexes);
    cudaFree(d_min_local_result);
    cudaFree(d_row_indexes);
    cudaDeviceSynchronize();
#endif
    const double elapsed = hpc_gettime() - tstart;

    print_solution(&tab);
    printf("Number of iterations: %d\n", it);
    fprintf(stderr,"\nExecution time %f seconds\n", elapsed);

#ifdef SERIAL
    free(tab.mat);
#else
    free(tab.mat);
    free(max_local_result);
    free(col_indexes);
    free(min_local_result);
    free(row_indexes);
#endif

    return 0;

}