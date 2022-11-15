/****************************************************************************
 *
 * opencl-cat-map.c - Arnold's cat map with OpenCL
 *
 * Copyright (C) 2016--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - La mappa del gatto di Arnold
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2021-12-04

Anche la mappa del gatto di Arnold è una vecchia conoscenza, che
abbiamo incontrato in una precedente esercitazione. In questo
esercizio si chiede di realizzare un programma OpenCL che trasforma
una immagine mediante la mappa del gatto. Riportiamo nel seguito la
descrizione del problema.

La [mappa del gatto di
Arnold](https://en.wikipedia.org/wiki/Arnold%27s_cat_map) è una
funzione che trasforma una immagine $P$ di dimensione $N \times N$ in
una nuova immagine $P'$ delle stesse dimensioni. Per ogni $0 \leq x <
N,\ 0 \leq y < N$, il pixel di coordinate $(x,y)$ in $P$ viene
collocato nella posizione $(x',y')$ di $P'$ dove:

$$
x' = (2x + y) \bmod N, \qquad y' = (x + y) \bmod N
$$

("mod" è l'operatore modulo, corrispondente all'operatore `%` del
linguaggio C). Si può assumere che le coordinate $(0, 0)$ indichino il
pixel in alto a sinistra e le coordinate $(N-1, N-1)$ quello in basso
a destra, in modo da poter indicizzare l'immagine come se fosse una
matrice in linguaggio C. La Figura 1 mostra graficamente la
trasformazione.

![Figura 1: La mappa del gatto di Arnold](cat-map.png)

La mappa del gatto ha proprietà sorprendenti. Applicata ad una
immagine ne produce una versione molto distorta. Applicata nuovamente
a quest'ultima immagine, ne produce una ancora più distorta, e così
via. Tuttavia, dopo un certo numero di iterazioni (il cui valore
dipende da $N$, ma che in ogni caso è sempre minore o uguale a $3N$)
ricompare l'immagine di partenza! (Figura 2).

![Figura 2: Alcune immagini ottenute iterando la mappa del gatto $k$ volte](cat-map-demo.png)

Il _tempo minimo di ricorrenza_ per l'immagine
[cat1368.pgm](cat1368.pgm) di dimensione $1368 \times 1368$ fornita
come esempio è $36$: iterando $k$ volte della mappa del gatto si
otterrà l'immagine originale se e solo se $k$ è multiplo di 36. Non è
nota alcuna formula analitica che leghi il tempo minimo di ricorrenza
alla dimensione $N$ dell'immagine.

Viene fornito un programma sequenziale che calcola la $k$-esima
iterata della mappa del gatto usando la CPU. Il programma viene
invocato specificando sulla riga di comando il numero di iterazioni
$k$. Il programma legge una immagine in formato PGM da standard input,
e produce una nuova immagine su standard output ottenuta applicando
$k$ volte la mappa del gatto. Occorre ricordarsi di redirezionare lo
standard output su un file, come indicato nelle istruzioni nel
sorgente.

Per sfruttare il parallelismo OpencL è utile organizzare i work-item
in una griglia bidimensionale. Ogni work-item calcola una singola
iterazione della mappa del gatto, copiando un pixel dell'immagine
corrente nella posizione appropriata della nuova immagine. La
segnatura del kernel sarà:

```C
__kernel void
cat_map_iter_kernel( __global const unsigned char *cur,
                     __global unsigned char *next,
                     int N )
```

(dove $N$ è la larghezza o altezza dell'immagine, che deve essere
quadrata). Utilizzando il proprio ID globale, ogni work-item determina
le coordinate $(x, y)$ del pixel su cui operare, e calcola le
coordinate $(x', y')$ del pixel dopo l'applicazione di una iterazione
della mappa del gatto. Per calcolare la $k$-esima iterata sarà quindi
necessario invocare il kernel $k$ volte, scambiando dopo ogni
iterazione le immagini corrente e successiva come fatto dal programma
seriale.

La soluzione precedente consente di parallelizzare il programma
fornito apportando minime modifiche. Si può anche definire un kernel
che calcoli direttamente la k-esima iterata della mappa del gatto con
una singola invocazione. La segnatura del nuovo kernel sarà

```C
__kernel void
cat_map_iter_k_kernel( __global const unsigned char *cur,
                       __global unsigned char *next,
                       int N, int k )
```

Come nel caso precedente, ciascun work-item determina le coordinate
$(x,y)$ del pixel di sua competenza. Le coordinate del pixel dopo $k$
iterazioni si possono ottenere applicando lo schema seguente:

```C
const int x = ...;
const int y = ...;
int xcur = x, ycur = y, xnext, ynext;

if ( x < N && y < N ) {
	while (k--) {
		xnext = (2*xcur + ycur) % N;
		ynext = (xcur + ycur) % N;
		xcur = xnext;
		ycur = ynext;
	}
	\/\* copia il pixel (x, y) dell'immagine corrente
	in posizione (xnext, ynext) della nuova immagine \*\/
}
```

In questo modo è sufficiente una singola invocazione del kernel
(anziché $k$ come nel caso precedente) per ottenere l'immagine
finale. Consiglio di misurare i tempi di esecuzione delle due
alternative per capire se e di quanto la seconda soluzione è più
efficiente della prima.

Per compilare:

        cc -std=c99 -Wall -Wpedantic opencl-cat-map.c simpleCL.c -o opencl-cat-map -lOpenCL

Per eseguire:

        ./opencl-cat-map k < input_file > output_file

Esempio:

        ./opencl-cat-map 100 < cat1368.pgm > cat1368.100.pgm

## File

- [opencl-cat-map.c](opencl-cat-map.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)
- [cat1368.pgm](cat1368.pgm) (il tempo di ricorrenza di questa immagine è 36)

***/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "simpleCL.h"

#include "pgmutils.h"

#ifndef SERIAL
sclKernel cat_map_iter_kernel;
sclKernel cat_map_iter_k_kernel;
#endif

/**
 * Compute the |k|-th iterate of the cat map for image |img|. The
 * width and height of the input image must be equal. This function
 * replaces the bitmap of |img| with the one resulting after ierating
 * |k| times the cat map. You need to allocate a temporary image, with
 * the same size of the original one, so that you read the pixel from
 * the "old" image and copy them to the "new" image (this is similar
 * to a stencil computation, as was discussed in class). After
 * applying the cat map to all pixel of the "old" image the role of
 * the two images is exchanged: the "new" image becomes the "old" one,
 * and vice-versa. At the end of the function, the temporary image
 * must be deallocated.
 */
void cat_map( PGM_image* img, int k )
{
    const int N = img->width;
    const size_t size = N * N * sizeof(img->bmap[0]);

#ifdef SERIAL
    /* [TODO] Modify the body of this function to allocate device memory,
       do the appropriate data transfer, and launch a kernel */
    unsigned char *cur = img->bmap;
    unsigned char *next = (unsigned char*)malloc( size );

    assert(next != NULL);
    for (int i=0; i<k; i++) {
        for (int y=0; y<N; y++) {
            for (int x=0; x<N; x++) {
                int xnext = (2*x+y) % N;
                int ynext = (x + y) % N;
                next[xnext + ynext*N] = cur[x+y*N];
            }
        }
        /* Swap old and new */
        unsigned char *tmp = cur;
        cur = next;
        next = tmp;
    }
    img->bmap = cur;
    free(next);
#else
    const sclDim block = DIM2(SCL_DEFAULT_WG_SIZE2D, SCL_DEFAULT_WG_SIZE2D);
    const sclDim grid = DIM2(sclRoundUp(N, SCL_DEFAULT_WG_SIZE2D),
                             sclRoundUp(N, SCL_DEFAULT_WG_SIZE2D));
    cl_mem d_cur, d_next;

    assert( img->width == img->height );

    /* Allocate bitmaps on the device */
    d_cur = sclMallocCopy(size, img->bmap, CL_MEM_READ_WRITE);
    d_next = sclMalloc(size, CL_MEM_READ_WRITE);

#if 0
    /* This version performs k kernel calls */
    fprintf(stderr, "Performing %d kernel calls\n", k);
    while( k-- ) {
        sclSetArgsEnqueueKernel(cat_map_iter_kernel,
                                grid, block,
                                ":b :b :d",
                                d_cur, d_next, N);
        cl_mem d_tmp = d_cur;
        d_cur = d_next;
        d_next = d_tmp;
    }
    sclMemcpyDeviceToHost(img->bmap, d_cur, size);
#else
    /* This version performs one kernel call */
    fprintf(stderr, "Performing a single kernel call\n");
    sclSetArgsEnqueueKernel(cat_map_iter_k_kernel,
                            grid, block,
                            ":b :b :d :d",
                            d_cur, d_next, N, k)
;
    sclMemcpyDeviceToHost(img->bmap, d_next, size);
#endif
    /* Free memory on device */
    sclFree(d_cur);
    sclFree(d_next);
#endif
}

int main( int argc, char* argv[] )
{
    PGM_image img;
    int niter;

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s niter < input_image > output_image\n", argv[0]);
        return EXIT_FAILURE;
    }
    niter = atoi(argv[1]);
    read_pgm(stdin, &img);
    if ( img.width != img.height ) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }
#ifndef SERIAL
    sclInitFromFile("opencl-cat-map.cl");
    cat_map_iter_kernel = sclCreateKernel("cat_map_iter_kernel");
    cat_map_iter_k_kernel = sclCreateKernel("cat_map_iter_k_kernel");
#endif
    const double tstart = hpc_gettime();
    cat_map(&img, niter);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "     Mpixels/sec : %f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);

    write_pgm(stdout, &img, "produced by opencl-cat-map.c");
    free_pgm(&img);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
