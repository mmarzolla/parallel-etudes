/****************************************************************************
 *
 * simd-map-levels.c -- Map gray levels on image
 *
 * Copyright (C) 2018--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Map gray levels on image
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-12-11

Let us consider a grayscale bitmap with $M$ rows and $N$ columns,
where the color of each pixel is encoded by an integer from 0 (black)
to 255 (white). Given two values _low, high_, $0 \leq \mathit{low} <
\mathit{high} \leq 255$, the function `map_levels(img, low, high)`
modifies `img` so that the pixels whose gray level is less than _low_
become black, those whose gray level is greater than _high_ become
white, and those whose gray level is between _low_ and _high_
(inclusive) are linearly mapped to the range $[0, 255]$.

Specifically, if $p$ is the gray level of a pixel, the function
computes the new gray level $p'$ as:

$$
p' = \begin{cases}
0 & \text{if}\ p < \mathit{low}\\
\displaystyle\frac{255 \times (p - \mathit{low})}{\mathit{high} - \mathit{low}} & \text{if}\ \mathit{low} \leq p \leq \mathit{high}\\
255 & \text{is}\ p > \mathit{high}
\end{cases}
$$

Figure 1 shows the image produced by the command

        ./simd-map-levels 100 255 < Yellow_palace_Winter.pgm > out.pgm

![Figure 1: Example of gray level mapping (_low_ = 100, _high_ = 255)](simd-map-levels.png)

As an example we provide the image [C1648109](C1648109.pgm) taken by
the [Voyager 1](https://voyager.jpl.nasa.gov/) probe on March 8,
1979. The image shows Io, one of the four [Galilean moons of the
planet Jupiter](https://en.wikipedia.org/wiki/Galilean_moons). The
Flight Engineer [Linda
Morabito](https://en.wikipedia.org/wiki/Linda_A._Morabito) was using
this image to uncover the background stars to determine the precise
location of the probe. To this aim, she needed to remap the levels so
that the faint stars in the background would be visible. This lead to
one of the most important discoveries of modern planetary sciences:
see by yourself by running the program

        ./simd-map-levels 10 30 < C1648109.pgm > out.pgm

and look at what appears next to the disc of Io at ten o'clock...

![Figure 2: Image C1648109 taken by Voyager 1 ([source](https://opus.pds-rings.seti.org/#/mission=Voyager&target=Io&cols=opusid,instrument,planet,target,time1,observationduration&widgets=mission,planet,target&order=time1,opusid&view=detail&browse=gallery&cart_browse=gallery&startobs=481&cart_startobs=1&detail=vg-iss-1-j-c1648109))](C1648109.png)

The file [simd-map-levels.c](simd-map-levels.c) contains a serial
implementation of function `map_levels()` above. The goal of this
exercise is to develop a SIMD version using GCC _vector datatypes_.
We start by defining a vector datatype `v4i` that represents four
integers:

```C
typedef int v4i __attribute__((vector_size(16)));
#define VLEN (sizeof(v4i)/sizeof(int))
```

The idea is to process the image four pixels at a time. However, the
serial code

```C
int *pixel = bmap + i*width + j;
if (*pixel < low)
    *pixel = BLACK;
else if (*pixel > high)
    *pixel = WHITE;
else
    *pixel = (255 * (*pixel - low)) / (high - low);
```

is problematic because it contains conditional statements that can not
be directly vectorized. To address this issue we use the _selection
and masking_ technique. Let `pixels` be a pointer to a `v4i` SIMD
array. Then, the expression `mask_black = (*pixels < low)` produces a
SIMD array of integers whose elements are -1 for those pixels whose
gray level is less than _low_, 0 otherwise. `mask_black` can therefore
be used as a bit mask to assign the correct values to these pixels.

Using the idea above, we can rewrite the code as follows:

```C
v4i *pixels = (v4i*)(bmap + i*width + j);
const v4i mask_black = (*pixels < low);
const v4i mask_white = (*pixels > high);
const v4i mask_map = ??? ;
*pixels = ( (mask_black & BLACK) |
            (mask_white & WHITE) |
            ( ??? ) );
```

Note that the compiler automatically promotes `BLACK` and `WHITE` to
SIMD vectors whose elements are all `BLACK` or `WHITE`,
respectively. (The code above can be further simplified since
`(mask_black & BLACK)` always produces a SIMD array whose elements are
all zeros: why?).

The SIMD version requires that

1. The bitmap is aligned in memory at a memory address that is
   multiple of 16;

2. The image width is an integer multiple of the `v4i` SIMD vector
   width, i.e., an integer multiple of 4.

Both conditions are satisfied: the program pads the image horizontally
so the next multiple of 4. Indeed, the attribute `width` of structure
`PGM_image` is the width of the _padded_ image, while `true_width` is
the true width of the _actual_ image.

To compile:

        gcc -std=c99 -Wall -Wpedantic -O2 -march=native simd-map-levels.c -o simd-map-levels

To execute:

        ./simd-map-levels low high < input_file > output_file

dove $0 \leq \mathit{low} < \mathit{high} \leq 255$.

Example:

        ./simd-map-levels 10 30 < C1648109.pgm > C1648109-map.pgm

## Files

- [simd-map-levels.c](simd-map-levels.c)
- [hpc.h](hpc.h)
- Some input images: [Yellow palace Winter](Yellow_palace_Winter.pgm), [C1648109.pgm](C1648109.pgm)

You can generate input images of arbitrary size with the command:

        convert -size 1024x768 plasma: -depth 8 test-image.pgm

***/

/* The following #define is required to make posix_memalign() visible */
#define _XOPEN_SOURCE 600

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef int v4i __attribute__((vector_size(16)));
#define VLEN (sizeof(v4i)/sizeof(int))

typedef struct {
    int width;   /* Padded width of the image (in pixels); this is a multiple of VLEN */
    int true_width; /* True width of the image (in pixels) */
    int height;  /* Height of the image (in pixels) */
    int maxgrey; /* Don't care (used only by the PGM read/write routines) */
    int *bmap;   /* buffer of width*height bytes; each element represents the gray level of a pixel (0-255) */
} PGM_image;

enum {
    BLACK = 0,
    WHITE = 255
};

/**
 * Read a PGM file from file `f`. Warning: this function is not
 * robust: it may fail on legal PGM images, and may crash on invalid
 * files since no proper error checking is done. The image width is
 * padded to the next integer multiple of 'VLEN`.
 */
void read_pgm( FILE *f, PGM_image* img )
{
    char buf[1024];
    const size_t BUFSIZE = sizeof(buf);
    char *s;

    assert(f != NULL);
    assert(img != NULL);

    /* Get the file type (must be "P5") */
    s = fgets(buf, BUFSIZE, f);
    if (0 != strcmp(s, "P5\n")) {
        fprintf(stderr, "Wrong file type %s\n", buf);
        exit(EXIT_FAILURE);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, BUFSIZE, f);
    } while (s[0] == '#');
    /* Get width, height */
    sscanf(s, "%d %d", &(img->true_width), &(img->height));
    /* Set `img->width` as the next integer multiple of `VLEN`
       greater than or equal to `img->true_width` */
    img->width = ((img->true_width + VLEN - 1) / VLEN) * VLEN;
    /* get maxgrey; must be less than or equal to 255 */
    s = fgets(buf, BUFSIZE, f);
    sscanf(s, "%d", &(img->maxgrey));
    if ( img->maxgrey > 255 ) {
        fprintf(stderr, "FATAL: maxgray=%d > 255\n", img->maxgrey);
        exit(EXIT_FAILURE);
    }
    /* The pointer img->bmap must be properly aligned to allow SIMD
       instructions, because the compiler emits SIMD instructions for
       aligned load/stores only. */
    int ret = posix_memalign((void**)&(img->bmap), __BIGGEST_ALIGNMENT__, (img->width)*(img->height)*sizeof(int));
    assert(0 == ret);
    assert(img->bmap != NULL);
    /* Get the binary data from the file */
    for (int i=0; i<img->height; i++) {
        for (int j=0; j<img->width; j++) {
            unsigned char c = 0;
            if (j < img->true_width) {
                const int nread = fscanf(f, "%c", &c);
                assert(nread == 1);
            }
            *(img->bmap + i*img->width + j) = c;
        }
    }
}

/**
 * Write the image `img` to file `f`; if not `NULL`, use the string
 * `comment` as metadata.
 */
void write_pgm( FILE *f, const PGM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->true_width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    for (int i=0; i<img->height; i++) {
        for (int j=0; j<img->true_width; j++) {
            fprintf(f, "%c", *(img->bmap + i*img->width + j));
        }
    }
}

/**
 * Free the bitmap associated with image `img`; note that the
 * structure pointed to by `img` is NOT deallocated; only `img->bmap`
 * is.
 */
void free_pgm( PGM_image *img )
{
    assert(img != NULL);
    free(img->bmap);
    img->bmap = NULL; /* not necessary */
    img->width = img->true_width = img->height = img->maxgrey = -1;
}

/*
 * Map the gray range [low, high] to [0, 255].
 */
void map_levels( PGM_image* img, int low, int high )
{
    const int width = img->width;
    const int height = img->height;
    int *bmap = img->bmap;
#ifdef SERIAL
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            int *pixel = bmap + i*width + j;
            if (*pixel < low)
                *pixel = BLACK;
            else if (*pixel > high)
                *pixel = WHITE;
            else
                *pixel = (255 * (*pixel - low)) / (high - low);
        }
    }
#else
    assert( width % VLEN == 0 );
    for (int i=0; i<height; i++) {
        for (int j=0; j<width-VLEN+1; j += VLEN) {
            v4i *pixels = (v4i*)(bmap + i*width + j);
            const v4i mask_black = (*pixels < low);
            const v4i mask_white = (*pixels > high);
            const v4i mask_map = ~(mask_black | mask_white);
            *pixels = ( (mask_black & BLACK) | /* can be omitted, is always {0, ... 0} */
                        (mask_white & WHITE) |
                        (mask_map & (255 * (*pixels - low)) / (high - low)));
        }
    }
#endif
}

int main( int argc, char* argv[] )
{
    PGM_image bmap;

    if ( argc != 3 ) {
        fprintf(stderr, "Usage: %s low high < in.pgm > out.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int low = atoi(argv[1]);
    const int high = atoi(argv[2]);
    if (low < 0 || low > 255) {
        fprintf(stderr, "FATAL: low=%d out of range\n", low);
        return EXIT_FAILURE;
    }
    if (high < 0 || high > 255 || high <= low) {
        fprintf(stderr, "FATAL: high=%d out of range\n", high);
        return EXIT_FAILURE;
    }
    read_pgm(stdin, &bmap);
    const double tstart = hpc_gettime();
    map_levels(&bmap, low, high);
    const double elapsed = hpc_gettime() - tstart;
    write_pgm(stdout, &bmap, "produced by simd-map-levels.c");
    fprintf(stderr, "Executon time (s): %f (%f Mops/s)\n", elapsed, (1e-6) * bmap.width * bmap.height / elapsed);
    free_pgm(&bmap);
    return EXIT_SUCCESS;
}
