/****************************************************************************
 *
 * simd-cat-map.c - Arnold's cat map
 *
 * Copyright (C) 2016--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% Ultimo aggiornamento: 2022-11-26

![Arnold's cat map](cat-map.png)

Scopo di questo esercizio è sviluppare una versione SIMD di una
funzione che calcola l'iterata della _mappa del gatto di Arnold_, una
vecchia conoscenza che abbiamo già incontrato in altre esercitazioni.
Riportiamo nel seguito la descrizione del problema.

La mappa del gatto trasforma una immagine $P$ di dimensione $N \times
N$ in una nuova immagine $P'$ delle stesse dimensioni. Per ogni $0
\leq x < N,\ 0 \leq y < N$, il pixel di coordinate $(x,y)$ in $P$
viene collocato nella posizione $(x',y')$ di $P'$ dove:

$$
x' = (2x + y) \bmod N, \qquad y' = (x + y) \bmod N
$$

("mod" è l'operatore modulo, corrispondente all'operatore `%` del
linguaggio C). Si può assumere che le coordinate $(0, 0)$ indichino il
pixel in alto a sinistra e le coordinate $(N-1, N-1)$ quello in basso
a destra, in modo da poter indicizzare l'immagine come se fosse una
matrice in linguaggio C. La Figura 1 mostra graficamente la
trasformazione.

![Figura 1: La mappa del gatto di Arnold](cat-map.svg)

La mappa del gatto ha proprietà sorprendenti. Applicata ad una
immagine ne produce una versione molto distorta. Applicata nuovamente
a quest'ultima immagine, ne produce una ancora più distorta, e così
via. Tuttavia, dopo un certo numero di iterazioni (il cui valore
dipende da $N$, ma che in ogni caso è sempre minore o uguale a $3N$)
ricompare l'immagine di partenza! (si veda la Figura 2).

![Figura 2: Alcune immagini ottenute iterando la mappa del gatto $k$ volte](cat-map-demo.png)

Il _tempo minimo di ricorrenza_ per l'immagine
[cat1368.pgm](cat1368.pgm) di dimensione $1368 \times 1368$ fornita
come esempio è $36$: iterando $k$ volte della mappa del gatto si
otterrà l'immagine originale se e solo se $k$ è multiplo di 36. Non è
nota alcuna formula analitica che leghi il tempo minimo di ricorrenza
alla dimensione $N$ dell'immagine.

Viene fornito un programma sequenziale che calcola la $k$-esima iterata
della mappa del gatto usando la CPU. Il programma viene invocato
specificando sulla riga di comando il numero di iterazioni $k$. Il
programma legge una immagine in formato PGM da standard input, e
produce una nuova immagine su standard output ottenuta applicando $k$
volte la mappa del gatto. Occorre ricordarsi di redirezionare lo
standard output su un file, come indicato nelle istruzioni nel
sorgente.  La struttura della funzione che calcola la k-esima iterata
della mappa del gatto è molto semplice:

```C
for (y=0; y<N; y++) {
	for (x=0; x<N; x++) {
		\/\* calcola le coordinate (xnew, ynew) del punto (x, y)
			dopo k applicazioni della mappa del gatto \*\/
		next[xnew + ynew*N] = cur[x+y*N];
	}
}
```

Per sfruttare il parallelismo SIMD possiamo ragionare come segue:
anziché calcolare le nuove coordinate di un punto alla volta,
calcoliamo le coordinate di quattro punti adiacenti $(x, y)$,
$(x+1,y)$, $(x+2,y)$, $(x+3,y)$ usando i _vector datatype_ del
compilatore. Per fare questo, definiamo le seguenti variabili di tipo
`v4i` (vettori SIMD di 4 interi):

- `vx`, `vy`: coordinate di quattro punti adiacenti, prima
  dell'applicazione della mappa del gatto;

- `vxnew`, `vynew`: nuova coordinate dei punti di cui sopra dopo
  l'applicazione della mappa del gatto.

Ricordiamo che il tipo `v4i` si definisce con `gcc` come

```C
	typedef int v4i __attribute__((vector_size(16)));
	#define VLEN (sizeof(v4i)/sizeof(int))
```

Posto $vx = \{x, x+1, x+2, x+3\}$, $vy = \{y, y, y, y\}$, possiamo
applicare ad essi le stesse operazioni aritmetiche applicate agli
scalari _x_ e _y_ per ottenere le nuove coordinate _vxnew_, _vynew_.
Fatto questo, al posto della singola istruzione:

```C
	next[xnew + ynew*N] = cur[x+y*N];
```

per spostare materialmente i pixel nella nuova posizione occorre
eseguire quattro istruzioni scalari:

```C
	next[vxnew[0] + vynew[0]*N] = cur[vx[0] + vy[0]*N];
	next[vxnew[1] + vynew[1]*N] = cur[vx[1] + vy[1]*N];
	next[vxnew[2] + vynew[2]*N] = cur[vx[2] + vy[2]*N];
	next[vxnew[3] + vynew[3]*N] = cur[vx[3] + vy[3]*n];
```

Si assuma che la dimensione $N$ dell'immagine sia sempre multipla di 4.

Compilare con:

        gcc -std=c99 -Wall -Wpedantic -O2 -march=native simd-cat-map.c -o simd-cat-map

Eseguire con:

        ./simd-cat-map k < input_file > output_file

Esempio:

        ./simd-cat-map 100 < cat368.pgm > cat1368-100.pgm

## Estensione

Le prestazioni della versione SIMD dell'algoritmo della mappa del
gatto dovrebbero risultare solo marginalmente migliori della versione
scalare (potrebbero addirittura essere peggiori). Analizzando il
codice assembly prodotto dal compilatore, si scopre che il calcolo del
modulo nelle due espressioni

```C
	vxnew = (2*vxold+vyold) % N;
	vynew = (vxold + vyold) % N;
```

viene realizzato usando operazioni scalari. Consultando la lista dei
_SIMD intrinsics_ sul [sito di
Intel](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
si scopre che non esiste una istruzione SIMD che realizzi la divisione
intera. Per migliorare le prestazioni del programma occorre quindi
ingegnarsi per calcolare i moduli senza fare uso della
divisione. Ragionando in termini scalari, osserviamo che se $0 \leq
xold < N$ e $0 \leq yold < N$, allora si ha necessariamente che $0
\leq 2 \times xold + yold < 3N$ e $0 \leq xold+yold < 2N$.

Pertanto, sempre in termini scalari, possiamo realizzare il calcolo di
`xnew` e `ynew` come segue:

```C
	xnew = (2*xold + yold);
	if (xnew >= N) { xnew = xnew - N; }
	if (xnew >= N) { xnew = xnew - N; }
	ynew = (xold + yold);
	if (ynew >= N) { ynew = ynew - N; }
```

Il codice precedente è meno leggibile della versione che usa
l'operatore modulo, ma ha il vantaggio di poter essere vettorizzato
ricorrendo al meccanismo di "selection and masking" visto a
lezione. Ad esempio, l'istruzione

```C
	if (xnew >= N) { xnew = xnew - N; }
```

può essere riscritta come

```C
	const v4i mask = (xnew >= N);
	xnew = (mask & (xnew - N)) | (mask & xnew);
```

che può essere ulteriormente semplificata come:

```C
	const v4i mask = (xnew >= N);
	xnew = xnew - (mask & N);
```

Si ottiene in questo modo un programma più complesso della versione
scalare, ma più veloce in quanto si riesce a sfruttare al meglio le
istruzioni SIMD.

Per compilare:

        gcc -std=c99 -Wall -Wpedantic -march=native -O2 simd-cat-map.c -o simd-cat-map

Per eseguire:

        ./simd-cat-map [niter] < in.pgm > out.pgm

Esempio:

        ./simd-cat-map 1024 < cat1368.pgm > cat1368-1024.pgm

## File

- [simd-cat-map.c](simd-cat-map.c)
- [hpc.h](hpc.h)
- [cat1368.pgm](cat1368.pgm) (il tempo di ricorrenza di questa immagine è 36)

 ***/

/* The following #define is required by posix_memalign(). It MUST
   be defined before including any other files. */
#define _XOPEN_SOURCE 600

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef int v4i __attribute__((vector_size(16)));
#define VLEN (sizeof(v4i)/sizeof(int))

#include "pgmutils.h"

/**
 * Compute the |k|-th iterate of the cat map for image |img|. You must
 * implement this function, starting with a serial version, and then
 * adding OpenMP pragmas.
 */
void cat_map( PGM_image* img, int k )
{
    const int N = img->width;
    unsigned char *cur = img->bmap;
    unsigned char *next;
    int x, y, i, ret;

    assert( img->width == img->height );
    assert( img->width % VLEN == 0);

    ret = posix_memalign((void**)&next, __BIGGEST_ALIGNMENT__, N*N*sizeof(*next));
    assert( 0 == ret );

#ifdef SERIAL
    for (y=0; y<N; y++) {
        /* [TODO] The SIMD version should compute the new position of
           four adjacent pixels (x,y), (x+1,y), (x+2,y), (x+3,y) using
           SIMD instructions. Assume that w (the image width) is
           always a multiple of VLEN. */
        for (x=0; x<N; x++) {
            int xold = x, xnew = xold;
            int yold = y, ynew = yold;
            for (i=0; i<k; i++) {
                xnew = (2*xold+yold) % N;
                ynew = (xold + yold) % N;
                xold = xnew;
                yold = ynew;
            }
            next[xnew + ynew*N] = cur[x+y*N];
        }
    }
#else
    /* SIMD version of the cat map iteration. The idea is to compute
       the new coordinates of four adjacent points (x, y), (x+1,y),
       (x+2,y) and (x+3,y) using SIMD instructions. To do so, we pack
       the coordinates into two SIMD registers vx = {x, x+1, x+2,
       x+3}, vy = {y, y, y, y}.

       Note that initially vx = {0, 1, 2, 3}, and at each iteration of
       the inner loop, all elements of vx are incremented by
       VLEN. Therefore, assuming VLEN == 4, at the second iteration we
       have vx = {4, 5, 6, 7}, then {8, 9, 10, 11} and so on. */
    for (y=0; y<N; y++) {
        v4i vx = {0, 1, 2, 3};
        const v4i vy = {y, y, y, y};
        for (x=0; x<N-VLEN+1; x += VLEN) {
            v4i xold = vx, xnew = xold;
            v4i yold = vy, ynew = yold;
            for (i=0; i<k; i++) {
#if 0
                xnew = (2*xold+yold) % N;
                ynew = (xold + yold) % N;
#else
                /* There is no SIMD instruction for integer division
                   in SSEx/AVX (_mm_div_epi32() exists on Intel MIC
                   only), so computing the remainder requires scalar
                   division and is extremely slow, so,
                   auto-vectorization does not really work in this
                   case.

                   The code below gets rid of the integer division, at
                   the cost of additional code complexity.  On my
                   Intel i7-4790 processor with gcc 7.5.0 the code
                   below is 10x faster than using the modulo operator
                   with auto-vectorization.

                   The situation is similar on ARM: on ARMv7 (armv7l,
                   Raspberry Pi4, gcc 8.3.0) there is no SIMD integer
                   division and the code below requires 66% the time
                   of the one using the modulo operator.
                */
                v4i mask;
                /* assuming 0 <= xold < N and 0 <= yold < N, we have
                   that 0 <= xnew < 3N; therefore, we might need to
                   subtract N at most twice from xnew to compute the
                   correct remainder modulo N. */
                xnew = (2*xold+yold);

                mask = (xnew >= N);
                xnew = (mask & (xnew - N)) | (~mask & xnew); // or: xnew = xnew - (mask & N);
                mask = (xnew >= N);
                xnew = (mask & (xnew - N)) | (~mask & xnew); // or: xnew = xnew - (mask & N);

                /* assuming 0 <= xold < N and 0 <= yold < N, we have
                   that 0 <= ynew < 2N; therefore, we might need to
                   subtract N at most once. */
                ynew = (xold + yold);
                mask = (ynew >= N);
                ynew = (mask & (ynew - N)) | (~mask & ynew); // or: ynew = ynew - (mask & N);
#endif
                xold = xnew;
                yold = ynew;
            }
            next[xnew[0] + ynew[0]*N] = cur[vx[0]+y*N];
            next[xnew[1] + ynew[1]*N] = cur[vx[1]+y*N];
            next[xnew[2] + ynew[2]*N] = cur[vx[2]+y*N];
            next[xnew[3] + ynew[3]*N] = cur[vx[3]+y*N];
            vx += VLEN;
        }
    }
#endif

    img->bmap = next;
    free(cur);
}

int main( int argc, char* argv[] )
{
    PGM_image img;
    int niter;

    if ( argc != 2 ) {
        fprintf(stderr, "Usage: %s niter < in.pgm > out.pgm\n\nExample: %s 684 < cat1368.pgm > out1368.pgm\n", argv[0], argv[0]);
        return EXIT_FAILURE;
    }
    niter = atoi(argv[1]);
    read_pgm(stdin, &img);
    if ( img.width != img.height ) {
        fprintf(stderr, "FATAL: width (%d) and height (%d) of the input image must be equal\n", img.width, img.height);
        return EXIT_FAILURE;
    }
    if ( img.width % VLEN ) {
        fprintf(stderr, "FATAL: this program expects the image width (%d) to be a multiple of %d\n", img.width, (int)VLEN);
        return EXIT_FAILURE;
    }
    const double tstart = hpc_gettime();
    cat_map(&img, niter);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "      SIMD width : %d bytes\n", (int)VLEN);
    fprintf(stderr, "      Iterations : %d\n", niter);
    fprintf(stderr, "    width,height : %d,%d\n", img.width, img.height);
    fprintf(stderr, "     Mpixels/sec : %f\n", 1.0e-6 * img.width * img.height * niter / elapsed);
    fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);

    write_pgm(stdout, &img, "produced by simd-cat-map.c");
    free_pgm(&img);
    return EXIT_SUCCESS;
}
