/****************************************************************************
 *
 * opencl-letters.c - Character counts
 *
 * Copyright (C) 2018--2024 Moreno Marzolla
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

/***
% HPC - Character counts
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-01-04

![By Willi Heidelbach, CC BY 2.5, <https://commons.wikimedia.org/w/index.php?curid=1181525>](letters.jpg)

The file [opencl-letters.c](opencl-letters.c) contains a serial
program that computes the number of occurrences of each lowercase
letter in an ASCII file read from standard input. The program is
case-insensitive, meaning that uppercase characters are treated as if
they were lowercase; non-letter characters are ignored. We provide
some substantial ASCII documents to experiment with, that have been
made available by the [Project Gutenberg](https://www.gutenberg.org/);
despite the fact that these documents have been written by different
authors, the frequencies of characters are quite similar. Indeed, it
is well known that the relative frequencies of characters are
language-dependent and more or less author-independent. You may
experiment with other free books in other languages that are available
on [Project Gutenberg Web site](https://www.gutenberg.org/).

In this exercise you are required to transform the program to make use
of OpenCL parallelism.

Compile with:

        gcc -std=c99 -Wall -Wpedantic opencl-letters.c simpleCL.c -o opencl-letters -lOpenCL

Run with:

        ./opencl-letters < the-war-of-the-worlds.txt

## Files

* [opencl-letters.c](opencl-letters.c) [hpc.h](hpc.h)
* [simpleCL.h](simpleCL.h) [simpleCL.c](simpleCL.c)
* [War and Peace](war-and-peace.txt) by L. Tolstoy
* [The Hound of the Baskervilles](the-hound-of-the-baskervilles.txt) by A. C. Doyle
* [The War of the Worlds](the-war-of-the-worlds.txt) by H. G. Wells

***/

/* The following #define is required by the implementation of
   hpc_gettime(). It MUST be defined before including any other
   file. */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
#include "hpc.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>

#include "simpleCL.h"

#define ALPHA_SIZE 26

#ifndef SERIAL
sclKernel hist_kernel;
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
    const sclDim BLOCK = DIM1(SCL_DEFAULT_WG_SIZE1D);
    const sclDim GRID = DIM1(sclRoundUp(len, SCL_DEFAULT_WG_SIZE1D));
    const size_t HIST_SIZE = ALPHA_SIZE * sizeof(hist[0]);

    for (int i=0; i<ALPHA_SIZE; i++) {
        hist[i] = 0;
    }
    /* Note: sclMallocCopy requires the second parameter to be
       non-const; this is actually required by the low-level OpenCL
       function clCreateBuffer. Hopefully it is ok to simply "cast
       away the const" */
    cl_mem d_text = sclMallocCopy(len+1, (char*)text, CL_MEM_READ_ONLY);
    cl_mem d_hist = sclMallocCopy(HIST_SIZE, hist, CL_MEM_READ_WRITE);

    sclSetArgsLaunchKernel(hist_kernel,
                           GRID, BLOCK,
                           ":b :d :b",
                           d_text, len, d_hist);

    sclMemcpyDeviceToHost(hist, d_hist, HIST_SIZE);
    sclFree(d_text);
    sclFree(d_hist);
    return 0;
#endif
}

/**
 * If `freq == 100`, draw `len` caracters; otherwise, draw a fraction
 * of `len` characters proportional to `freq`.
 */
void bar( float freq, int len )
{
    for (int i=0; i<len*freq/100; i++) {
        printf("#");
    }
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
        const float freq = 100.0*hist[i]/nlet;
        printf("%c : %8d (%6.2f%%) ", 'a'+i, hist[i], freq);
        bar(freq, 65);
        printf("\n");
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
#ifndef SERIAL
    sclInitFromFile("opencl-letters.cl");
    hist_kernel = sclCreateKernel("hist_kernel");
#endif

    const double tstart = hpc_gettime();
    make_hist(text, hist);
    const double elapsed = hpc_gettime() - tstart;
    print_hist(hist);
    fprintf(stderr, "Elapsed time: %f\n", elapsed);

    free(text);

    sclFinalize();

    return EXIT_SUCCESS;
}
