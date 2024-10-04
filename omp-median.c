/****************************************************************************
 *
 * omp-median.c - Image denoising using a median filter
 *
 * Copyright (C) 2018--2023 Moreno Marzolla <moreno.marzolla@unibo.it>
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
% HPC - Image denoising using a median filter
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-11-03

The file [omp-median.c](omp-median.c) contains a serial implementation
of an _image denoising_ algorithm that can be used to clean color
images. The algorithm replaces the color of each pixel with the
_median_ of a neighborhood of radius `RADIUS` (including itself).

To compile:

        gcc -std=c99 -fopenmp -Wall -Wpedantic omp-median.c -o omp-median

To execute:

        ./omp-median inputfile outputfile

## Files

- [omp-median.c](omp-median.c)

 ***/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <omp.h>

typedef uint16_t data_t;

int IDX(int i, int j, int cols, int rows)
{
#ifdef REPLICATE
    i = (i<0 ? 0 : (i>=rows ? rows-1 : i));
    j = (j<0 ? 0 : (j>=cols ? cols-1 : j));
#else
    i = (i + rows) % rows;
    j = (j + cols) % cols;
#endif
    return (i*cols + j);
}

void swap(data_t *v, int i, int j)
{
    const data_t tmp = v[i];
    v[i] = v[j];
    v[j] = tmp;
}

int partition(data_t *v, int start, int end)
{
    /* L'invariante della procedura partition() descritta nel libro
       è la seguente:

       - v[k] <= pivot per ogni start <= k <= i
       - v[k] > pivot per ogni  i+1 <= k < j

    */
    const data_t pivot = v[end];
    int i = (start - 1), j;

    for (j = start; j < end; j++) {
        if (v[j] <= pivot) {
            i++;
            swap(v, i, j);
        }
    }

    swap(v, i+1, end);
    return i + 1;
}

int quickselect_rec(data_t *v, int start, int end, int k)
{
    assert(start <= end);
    const int split = partition(v, start, end);
    if (k == split)
        return v[k];
    else if (k < split)
        return quickselect_rec(v, start, split - 1, k);
    else
        return quickselect_rec(v, split + 1, end, k);
}

data_t median(data_t *v, int n)
{
    return quickselect_rec(v, 0, n-1, n/2);
}

void median_filter( int radius, data_t *bmap, int width, int height )
{
    const size_t tmp_len = (2*radius+1)*(2*radius+1);
    data_t *out = (data_t*)malloc(width*height*sizeof(data_t));
    assert(out != NULL);

#pragma omp parallel default(none) shared(width, height, bmap, out, radius, tmp_len)
    {
        data_t *tmp = (data_t*)malloc(tmp_len*sizeof(data_t));
        assert(tmp != NULL);
#pragma omp for collapse(2)
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                int k = 0;
                for (int di=-radius; di<=radius; di++) {
                    for (int dj=-radius; dj<=radius; dj++) {
                        tmp[k++] = bmap[IDX(i+di, j+dj, width, height)];
                    }
                }
                assert(k == tmp_len);
                out[IDX(i, j, width, height)] = median(tmp, tmp_len);
            }
        }
        free(tmp);
    }
    memcpy(bmap, out, width*height);
    free(out);
}

int main( int argc, char *argv[] )
{
    const int RADIUS = 51;
    const int WIDTH = 200;
    const int HEIGHT = 200;
    const size_t IMG_SIZE = WIDTH * HEIGHT * sizeof(data_t);

    if (argc != 3) {
        fprintf(stderr, "Usage: %s filein fileout\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE* filein = fopen(argv[1], "r");
    if (filein == NULL) {
        fprintf(stderr, "FATAL: can not open \"%s\"\n", argv[1]);
        return EXIT_FAILURE;
    }

    data_t *img = (data_t*)malloc(IMG_SIZE); assert(img != NULL);
    const size_t nread = fread(img, WIDTH*HEIGHT, sizeof(data_t), filein);
    assert(nread == WIDTH*HEIGHT);
    fclose(filein);

    const double tstart = omp_get_wtime();
    median_filter(RADIUS, img, WIDTH, HEIGHT);
    const double elapsed = omp_get_wtime() - tstart;
    fprintf(stderr, "Execution time: %f\n", elapsed);

    FILE* fileout = fopen(argv[2], "w");
    if (fileout == NULL) {
        fprintf(stderr, "FATAL: can not create \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }

    const size_t nwritten = fwrite(img, WIDTH*HEIGHT, sizeof(data_t), fileout);
    assert(nwritten == WIDTH*HEIGHT);
    fclose(fileout);

    free(img);

    return EXIT_SUCCESS;
}
