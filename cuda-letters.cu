/****************************************************************************
 *
 * cuda-letters.cu - Character counts
 *
 * Copyright (C) 2018--2024 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Character counts
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2024-09-03

![By Willi Heidelbach, CC BY 2.5, <https://commons.wikimedia.org/w/index.php?curid=1181525>](letters.jpg)

The file [cuda-letters.cu](cuda-letters.cu) contains a serial program
that computes the number of occurrences of each lowercase letter in an
ASCII file read from standard input. The program is case-insensitive,
meaning that uppercase characters are treated as if they were
lowercase; non-letter characters are ignored. We provide some
substantial ASCII documents to experiment with, that have been made
available by the [Project Gutenberg](https://www.gutenberg.org/);
despite the fact that these documents have been written by different
authors, the frequencies of characters are quite similar. Indeed, it
is well known that the relative frequencies of characters are
language-dependent and more or less author-independent. You may
experiment with other free books in other languages that are available
on [Project Gutenberg Web site](https://www.gutenberg.org/).

In this exercise you are required to transform the program to make use
of CUDA parallelism.

Compile with:

        nvcc cuda-letters.cu -o cuda-letters

Run with:

        ./cuda-letters < the-war-of-the-worlds.txt

## Files

* [cuda-letters.cu](cuda-letters.cu)
* [War and Peace](war-and-peace.txt) by L. Tolstoy
* [The Hound of the Baskervilles](the-hound-of-the-baskervilles.txt) by A. C. Doyle
* [The War of the Worlds](the-war-of-the-worlds.txt) by H. G. Wells

***/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"

#define ALPHA_SIZE 26
#ifndef SERIAL
#define BLKDIM 1024

__global__ void
hist_kernel( const char *text,
             int len,
             int *hist )
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int li = threadIdx.x;
    __shared__ int local_hist[ALPHA_SIZE];

    /* reset local histogram */
    if (li < ALPHA_SIZE)
        local_hist[li] = 0;

    __syncthreads();

    if (i < len) {
        char c = text[i];

        if (c >= 'A' && c <= 'Z')
            c = (c - 'A') + 'a';

        if (c >= 'a' && c <= 'z')
            atomicAdd(&local_hist[ c - 'a' ], 1);
    }

    __syncthreads();

    if (li < ALPHA_SIZE)
        atomicAdd(&hist[li], local_hist[li]);
}
#endif

/**
 * Count occurrences of letters 'a'..'z' in `text`; uppercase
 * characters are transformed to lowercase, and all other symbols are
 * ignored. `text` must be zero-terminated. `hist` will be filled with
 * the computed counts. Returns the total number of letters found.
 */
int make_hist( const char *text, int hist[ALPHA_SIZE] )
{
    const size_t len = strlen(text);
#ifdef SERIAL
    int nlet = 0; /* total number of alphabetic characters processed */
    /* [TODO] Parallelize this function */

    /* Reset the histogram */
    for (int j=0; j<ALPHA_SIZE; j++) {
        hist[j] = 0;
    }
    /* Count occurrences */
    for (int i=0; i < len; i++) {
        const char c = text[i];
        if (isalpha(c)) {
            nlet++;
            hist[ tolower(c) - 'a' ]++;
        }
    }
    return nlet;
#else
    const dim3 BLOCK(BLKDIM);
    const dim3 GRID((len + BLKDIM-1)/BLKDIM);
    const size_t HIST_SIZE = ALPHA_SIZE * sizeof(hist[0]);
    char *d_text;
    int *d_hist;

    for (int i=0; i<ALPHA_SIZE; i++) {
        hist[i] = 0;
    }
    cudaSafeCall( cudaMalloc( (void**)&d_text, len+1) );
    cudaSafeCall( cudaMemcpy( d_text, text, len+1, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMalloc( (void**)&d_hist, HIST_SIZE) );
    cudaSafeCall( cudaMemcpy( d_hist, hist, HIST_SIZE, cudaMemcpyHostToDevice) );

    hist_kernel<<< GRID, BLOCK >>>(d_text, len, d_hist); cudaCheckError();

    cudaSafeCall( cudaMemcpy(hist, d_hist, HIST_SIZE, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaFree(d_text) );
    cudaSafeCall( cudaFree(d_hist) );
    return 0;
#endif
}

/**
 * Print frequencies
 */
void print_hist( int hist[ALPHA_SIZE] )
{
    int i;
    int nlet = 0;
    for (i=0; i<ALPHA_SIZE; i++) {
        nlet += hist[i];
    }
    for (i=0; i<ALPHA_SIZE; i++) {
        printf("%c : %8d (%6.2f%%)\n", 'a'+i, hist[i], 100.0*hist[i]/nlet);
    }
    printf("    %8d total\n", nlet);
}

int main( void )
{
    int hist[ALPHA_SIZE] = { 0 };
    const size_t size = 5*1024*1024; /* maximum text size: 5 MB */
    char *text = (char*)malloc(size); assert(text != NULL);

    const size_t len = fread(text, 1, size-1, stdin);
    text[len] = '\0'; /* put a termination mark at the end of the text */

    const double tstart = hpc_gettime();
    make_hist(text, hist);
    const double elapsed = hpc_gettime() - tstart;
    print_hist(hist);
    fprintf(stderr, "Elapsed time: %f\n", elapsed);

    free(text);

    return EXIT_SUCCESS;
}