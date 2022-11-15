/******************************************************************************
 *
 * opencl-coupled-oscillators.c - One-dimensional coupled oscillators system
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 ******************************************************************************/

/***
% HPC - Oscillatori accoppiati
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2022-03-18

![Un gruppo di metronomi accoppiati tende a sincronizzarsi anche se il periodo naturale di ciascuno è diverso (Fonte: [Università di Harvard](https://sciencedemonstrations.fas.harvard.edu/presentations/synchronization-metronomes))](coupled_metronomes.jpg)

Consideriamo $n$ punti di massa $m$ disposti lungo una retta alle
coordinate $x_0, x_1, \ldots, x_{n-1}$. Masse adiacenti sono collegate
da una molla di costante elastica $k$ e lunghezza a riposo $L$. Il
primo e l'ultimo punto (quelli in posizione $x_0$ e $x_{n-1}$ occupano
una posizione fissa e non possono muoversi.

![Figura 1: Oscillatori accoppiati](opencl-coupled-oscillators.png)

Se all'istante iniziale una delle molle non è a riposo, si innescano
delle oscillazioni che, in mancanza di attrito, proseguiranno
indefinitamente. Sfruttando la seconda legge della dinamica di Newton
$F = ma$ e la legge di Hooke che afferma che una molla di costante $k$
compressa di una quantità $\Delta x$ esercita una forza $k \Delta x$,
sviluppiamo un programma che, date le posizioni e le velocità iniziali
delle masse, calcoli posizioni e velocità al tempo $t > 0$. Il
programma si basa su un algoritmo iterativo che partendo dalle
posizioni e velocità delle masse al tempo $t$, determina le nuove
posizioni e le nuove velocità al tempo $t + \Delta t$. In particolare,
la funzione

```C
step(double *x, double *v, double *xnext, double *vnext, int n)
```

calcola posizione `xnext[i]` e velocità `vnext[i]` della massa
$i$-esima al tempo $t + \Delta t$, $0 \le i < n$, date le posizioni
`x[i]` e velocità `v[i]` al tempo $t$.

1. Per ogni $i = 1, \ldots, n-2$ si calcola la forza $F_i$ che agisce
   sulla massa $i$-esima come $F_i := k \times (x_{i-1} -2x_i +
   x_{i+1})$; si noti come la forza non dipenda dalla lunghezza $L$
   delle molle a riposo. Le masse 0 e $n-1$ rimangono in posizione
   fissa, quindi le forze che agiscono su di esse possono essere
   omesse.

2. Per ogni $i = 1, \ldots, n-2$ si calcola la nuova velocità $v'_i$
   della massa $i$-esima al tempo $t + \Delta t$ come $v'_i := v_i +
   (F_i / m) \Delta t$. Le masse 0 e $n-1$ restano fisse, quindi la
   loro velocità sarà sempre zero.

3. Per ogni $i = 1, \ldots, n-2$ si calcola la nuova posizione $x'i$
   della massa $i$-esima al tempo $t + \Delta t$ come $x'_i := x_i +
   v'_i \Delta t$. Le masse 0 e $n-1$ restano ferme quindi la loro
   posizione al tempo $t + \Delta t$ sarà uguale a quella al tempo
   $t$: $x'_0 := x_0$, $x'_{n-1} := x_{n-1}$.

Il file [opencl-coupled-oscillators.c](opencl-coupled-oscillators.c)
contiene una versione seriale del programma che calcola l'evoluzione
di un insieme di oscillatori accoppiati. Il programma produce una
immagine bidimensionale in cui ogni riga mostra le energie potenziali
delle $n-1$ molle in ogni istante di tempo. Al termine dell'esecuzione
il programma dovrebbe produrre un file `coupled-oscillators.ppm`
contenente una immagine simile alla Figura 2:

![Figura 2: energia potenziale delle molle](coupled-oscillators.png)

Parallelizzare la funzione `step()` trasformandola (in tutto o in
parte) in un kernel OpenCL.

Per compilare:

        cc opencl-coupled-oscillators.c sipleCL.c -o opencl-coupled-oscillators -lm -lOpenCL

Per eseguire:

        ./opencl-coupled-oscillators [N]

Esempio:

        ./opencl-coupled-oscillators 1024

## File

- [opencl-coupled-oscillators.c](opencl-coupled-oscillators.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

 ***/
#include "hpc.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

/* Number of initial steps to skip, before starting to take pictures */
#define TRANSIENT 50000
/* Number of steps to record in the picture */
#define NSTEPS 800

/* Some physical constants; note that these are defined as symbolic
   values rather than constants, since they must be visible inside a
   kernel functions (and normal constants are not, unless they are
   stored in constant memory on the device) */
/* Integration time step */
#define dt 0.02f
/* spring constant (large k = stiff spring, small k = soft spring) */
#define k 0.2f
/* mass */
#define m 1.0f
/* Length of each spring at rest */
#define L 1.0f

/* Initial conditions: all masses are evenly placed so that the
   springs are at rest; some of the masses are displaced to start the
   movement. */
void init( float *x, float *v, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i*L;
        v[i] = 0.0;
    }
    /* displace some of the masses */
    x[n/3  ] -= 0.5*L;
    x[n/2  ] += 0.7*L;
    x[2*n/3] -= 0.7*L;
}

/**
 * Performs one simulation step: starting from the current positions
 * `x[]` and velocities `v[]` of the masses, compute the next
 * positions `xnext[]` and velocities `vnext[]`.
 */
#ifdef SERIAL
void step( float *x, float *v, float *xnext, float *vnext, int n )
{
    int i;
    for (i=0; i<n; i++) {
        if ( i > 0 && i < n - 1 ) {
            /* Compute the net force acting on mass i */
            const float F = k*(x[i-1] - 2*x[i] + x[i+1]);
            const float a = F/m;
            /* Compute the next position and velocity of mass i */
            vnext[i] = v[i] + a*dt;
            xnext[i] = x[i] + vnext[i]*dt;
        } else {
            xnext[i] = x[i];
            vnext[i] = 0.0;
        }
    }
}
#endif

/**
 * Compute x*x
 */
float squared(float x)
{
    return x*x;
}

/**
 * Compute the maximum energy among all springs.
 */
float maxenergy(const float *x, int n)
{
    int i;
    float maxenergy = -INFINITY;
    for (i=1; i<n; i++) {
        maxenergy = fmaxf(0.5*k*squared(x[i]-x[i-1]-L), maxenergy);
    }
    return maxenergy;
}

void dumpenergy(FILE *fout, const float *x, int n, float maxen)
{
    int i;
    /* Dump spring energies (light color = high energy) */
    maxen = maxenergy(x, n);
    for (i=1; i<n; i++) {
        const float displ = x[i] - x[i-1] - L;
        const float energy = 0.5*k*squared(displ);
        const float v = fminf(energy/maxen, 1.0);
        fprintf(fout, "%c%c%c", 0, (int)(255*v*(displ<0)), (int)(255*v*(displ>0)));
    }
}

int main( int argc, char *argv[] )
{
    int s, cur = 0, next;
    float maxen;
    int N = 1024;
    const char* fname = "coupled-oscillators.ppm";
#ifdef SERIAL
    float *x[2], *v[2];
#else
    float *x, *v;
    cl_mem d_x[2], d_v[2];
#endif

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [N]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (2 == argc) {
        N = atoi(argv[1]);
    }

    const size_t size = N * sizeof(float);

    FILE *fout = fopen(fname, "w");
    if (NULL == fout) {
        printf("Cannot open %s for writing\n", fname);
        return EXIT_FAILURE;
    }

    /* Write the header of the output file */
    fprintf(fout, "P6\n");
    fprintf(fout, "%d %d\n", N-1, NSTEPS);
    fprintf(fout, "255\n");

#ifdef SERIAL
    x[0] = (float*)malloc(size); assert(x[0]);
    x[1] = (float*)malloc(size); assert(x[1]);
    v[0] = (float*)malloc(size); assert(v[0]);
    v[1] = (float*)malloc(size); assert(v[1]);
    init(x[cur], v[cur], N);
#else
    sclInitFromFile("opencl-coupled-oscillators.cl");
    sclKernel step_kernel = sclCreateKernel("step_kernel");
    sclDim block, grid;
    sclWGSetup1D(N, &grid, &block);

    /* Allocate host copies of x and v */
    x = (float*)malloc(size); assert(x);
    v = (float*)malloc(size); assert(v);
    init(x, v, N);

    /* Allocate device copies of x and v */
    d_x[cur] = sclMallocCopy(size, x, CL_MEM_READ_WRITE);
    d_x[1-cur] = sclMalloc(size, CL_MEM_READ_WRITE);
    d_v[cur] = sclMallocCopy(size, v, CL_MEM_READ_WRITE);
    d_v[1-cur] = sclMalloc(size, CL_MEM_READ_WRITE);
#endif

    /* Write NSTEPS rows in the output image */
    for (s=0; s<TRANSIENT + NSTEPS; s++) {
        next = 1 - cur;
#ifdef SERIAL
        step(x[cur], v[cur], x[next], v[next], N);
#else
        sclSetArgsEnqueueKernel(step_kernel,
                                grid, block,
                                ":b :b :b :b :f :f :f :d",
                                d_x[cur], d_v[cur], d_x[next], d_v[next], k, m, dt, N);
#endif
        if (s >= TRANSIENT) {
#ifdef SERIAL
            if (s == TRANSIENT) {
                maxen = maxenergy(x[next], N);
            }
            dumpenergy(fout, x[next], N, maxen);
#else
            sclMemcpyDeviceToHost(x, d_x[next], size);
            if (s == TRANSIENT) {
                maxen = maxenergy(x, N);
            }
            dumpenergy(fout, x, N, maxen);
#endif
        }
        cur = 1 - cur;
    }

#ifdef SERIAL
    free(x[0]);
    free(x[1]);
    free(v[0]);
    free(v[1]);
#else
    free(x);
    free(v);
    sclFree(d_x[0]);
    sclFree(d_x[1]);
    sclFree(d_v[0]);
    sclFree(d_v[1]);
    sclFinalize();
#endif

    fclose(fout);
    return EXIT_SUCCESS;
}
