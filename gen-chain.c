/******************************************************************************
 * Se nel programma SSSP si parallelizza il ciclo "for" piu' esterno
 * il programma potrebbe sembrare funzionante sui grafi di esempio. In
 * realta' il programma non funziona: questo file genera un input che
 * rappresenta un grafo con 1000 nodi e 999 archi, in cui i nodi sono
 * collegati come in una catena. Con tale input il programma
 * omp-sssp.c parallelizzato in modo errato fallisce (abbastanza)
 * regolarmente, nel senso che calcola risultati diversi rispetto alla
 * versione corretta.
 *
 * Compilare con: gcc gen-chain.c -o gen-chain
 *
 * Eseguire con: ./gen-chain > chain.gr
 ******************************************************************************/
#include <stdio.h>

int main( void )
{
    const int n = 1000;
    int i;
    printf("p sp %d %d\n", n, n-1);
    for (i=1; i<n; i++) {
        printf("a %d %d %d\n", i, i+1, 1);
    }
    return 0;
}
