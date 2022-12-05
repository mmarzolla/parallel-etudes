/****************************************************************************
 *
 * simd-map-levels.c -- Map gray levels on image
 *
 * Copyright (C) 2018--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Mappatura livelli di grigio
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2022-11-26

Consideriamo una immagine bitmap a toni di grigio di $M$ righe e $N$
colonne, in cui il colore di ogni pixel sia codificata con un intero
da 0 (nero) a 255 (bianco). Dati due valori interi _low, high_ con $0
\leq \mathit{low} < \mathit{high} \leq 255$, la funzione
`map_levels(img, low, high)` modifica l'immagine `img` rendendo neri
tutti i pixel il cui livello di grigio è minore di _low_, bianchi
tutti quelli il cui livello di grigio è maggiore di _high_, e mappando
i rimanenti nell'intervallo $[0, 255]$. Più in dettaglio, detto $p$ il
livello di grigio di un pixel, la funzione deve calcolare il nuovo
livello $p'$ come:

$$
p' = \begin{cases}
0 & \text{se}\ p < \mathit{low}\\
\displaystyle\frac{255 \times (p - \mathit{low})}{\mathit{high} - \mathit{low}} & \text{se}\ \mathit{low} \leq p \leq \mathit{high}\\
255 & \text{se}\ p > \mathit{high}
\end{cases}
$$

La Figura 1 mostra un esempio ottenuto con il comando

        ./simd-map-levels 100 255 < Yellow_palace_Winter.pgm > out.pgm

![Figura 1: Mappatura di livelli di grigio (_low_ = 100, _high_ = 255)](simd-map-levels.png)

Come caso di studio reale, viene fornita l'immagine
[C1648109](C1648109.pgm) ripresa dalla sonda [Voyager
1](https://voyager.jpl.nasa.gov/) l'8 marzo 1979, che mostra Io, uno
dei quattro [satelliti
Galileiani](https://en.wikipedia.org/wiki/Galilean_moons) di
Giove. L'ingegnera di volo [Linda
Morabito](https://it.wikipedia.org/wiki/Linda_Morabito) era
interessata ad evidenziare le stelle sullo sfondo per ottenere la
posizione precisa della sonda. A tale scopo ha rimappato i livello di
grigio, facendo una delle scoperte più importanti della
missione. Provate ad applicare il programma
[simd-map-levels.c](simd-map-levels.c) all'immagine ponendo _low_ = 10
e _high_ = 30, e osservate cosa compare a ore dieci accanto al disco
di Io...

![Figura 2: Immagine C1648109 di Io ripresa da Voyager 1 ([fonte](https://opus.pds-rings.seti.org/#/mission=Voyager&target=Io&cols=opusid,instrument,planet,target,time1,observationduration&widgets=mission,planet,target&order=time1,opusid&view=detail&browse=gallery&cart_browse=gallery&startobs=481&cart_startobs=1&detail=vg-iss-1-j-c1648109))](C1648109.png)

Il file [simd-map-levels.c](simd-map-levels.c) contiene una
implementazione seriale dell'operatore per rimappare i livelli di
grigio. Scopo di questo esercizio è svilupparne una versione SIMD
utilizzando i _vector datatype_ del compilatore GCC. Ogni pixel
dell'immagine è codificato da un valore di tipo `int` per evitare
problemi di _overflow_ durante le operazioni aritmetiche. Definiamo un
tipo `v4i` per rappresentare un vettore SIMD composto da 4 `int`:

```C
typedef int v4i __attribute__((vector_size(16)));
#define VLEN (sizeof(v4i)/sizeof(int))
```

L'idea è di elaborare l'immagine a blocchi di 4 pixel adiacenti. Per
vettorizzare il codice seriale

```C
int *pixel = bmap + i*width + j;
if (*pixel < low)
    *pixel = BLACK;
else if (*pixel > high)
    *pixel = WHITE;
else
    *pixel = (255 * (*pixel - low)) / (high - low);
```

occorre eliminare le strutture condizionali "if".  Definiamo una
variabile `pixels` come puntatore ad un array SIMD. L'espressione
`mask_black = (*pixels < low)` produce come risultato un nuovo array
SIMD i cui elementi valgono -1 in corrispondenza dei pixel con valore
minore alla soglia, e 0 per gli altri pixel. Il codice sopra può
quindi essere riscritto come:

```C
v4i *pixels = (v4i*)(bmap + i*width + j);
const v4i mask_black = (*pixels < low);
const v4i mask_white = (*pixels > high);
const v4i mask_map = ??? ;
*pixels = ( (mask_black & BLACK) |
            (mask_white & WHITE) |
            ( ??? ) );
```

Si noti che `BLACK` e `WHITE` vengono automaticamente promossi dal
compilatore ad array SIMD i cui elementi valgono tutti `BLACK` e
`WHITE`, rispettivamente.

Il frammento di codice precedente può essere semplificato perché
l'espressione `(mask_black & BLACK)` produce sempre un array SIMD i
cui elementi sono tutti zero (perché?).

Per funzionare correttamente la versione SIMD richiede che:

1. La bitmap sia allocata a partire da un indirizzo di memoria
   multiplo di 16;

2. La larghezza dell'immagine sia multipla dell'ampiezza di un
   registro SIMD (4, se usiamo il tipo `v4i`)

Entrambe le condizioni sono soddisfatte nel programma e nelle immagini fornite.

Compilare con:

        gcc -std=c99 -Wall -Wpedantic -O2 -march=native simd-map-levels.c -o simd-map-levels

Eseguire con:

        ./simd-map-levels low high < input_file > output_file

dove $0 \leq \mathit{low} < \mathit{high} \leq 255$.

Esempio:

        ./simd-map-levels 10 30 < C1648109.pgm > C1648109-map.pgm

## File

- [simd-map-levels.c](simd-map-levels.c)
- [hpc.h](hpc.h)
- Immagini di esempio: [Yellow palace Winter](Yellow_palace_Winter.pgm), [C1648109.pgm](C1648109.pgm)

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
    int width;   /* Width of the image (in pixels) */
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
 * files since no proper error checking is done.
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
    sscanf(s, "%d %d", &(img->width), &(img->height));
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
            unsigned char c;
            const int nread = fscanf(f, "%c", &c);
            assert(nread == 1);
            *(img->bmap + i*img->width + j) = c;
        }
    }
}

/**
 * Write the image `img` to file `f`; if not NULL, use the string
 * `comment` as metadata.
 */
void write_pgm( FILE *f, const PGM_image* img, const char *comment )
{
    assert(f != NULL);
    assert(img != NULL);

    fprintf(f, "P5\n");
    fprintf(f, "# %s\n", comment != NULL ? comment : "");
    fprintf(f, "%d %d\n", img->width, img->height);
    fprintf(f, "%d\n", img->maxgrey);
    for (int i=0; i<img->height; i++) {
        for (int j=0; j<img->width; j++) {
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
    img->width = img->height = img->maxgrey = -1;
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
    if ( bmap.width % VLEN ) {
        fprintf(stderr, "FATAL: the image width (%d) must be multiple of %d\n", bmap.width, (int)VLEN);
        return EXIT_FAILURE;
    }
    const double tstart = hpc_gettime();
    map_levels(&bmap, low, high);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Executon time (s): %f\n", elapsed);
    write_pgm(stdout, &bmap, "produced by simd-map-levels.c");
    free_pgm(&bmap);
    return EXIT_SUCCESS;
}
