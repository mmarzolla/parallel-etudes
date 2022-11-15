# Parallel Etudes

This repository contains the source code of exercises that are used in
the lab sessione for the High Performance Computing (HPC) course,
University of Bologna.

## Prerequisites

To build the executables and documentation for the programs, the
following tools are required:

- [Pandoc](https://pandoc.org/)

- [Sed](https://www.gnu.org/software/sed/)

- [GNU make](https://www.gnu.org/software/make/)

- [unifdef](https://dotat.at/prog/unifdef/)

## How it works

Sono presenti un certo numero di sorgenti in C e alcuni header file. I
sorgenti includono all'inizio un blocco di testo delimitato da

```C
/***

...

***/
```

Il Makefile estrae tramite `sed` tutto quello che è contenuto tra i
due delimitatori, lo interpreta usando `markdown` e lo converte in
HTML utilizzando [pandoc](https://pandoc.org/index.html). Markdown è
un linguaggio di markup minimale, che consente di scrivere testo ASCII
seguendo una sintassi molto semplice e "permissiva", e convertire tale
testo in diversi formati tra cui HTML.

Partendo da ciascun file `.c` vengono generate due versioni, una
contenente lo scheletro di programma da completare, e una contenente
la soluzione.  Per decidere cosa va in una versione o nell'altra si
può utilizzare il simbolo di preprocessore `HANDOUT`. Il Makefile
utilizza il programma [unifdef](https://dotat.at/prog/unifdef/) per
generare lo scheletro da completare definendo `HANDOUT`, e la soluzione
non definendo tale simbolo. Quindi, se abbiamo un pezzo di codice che
vogliamo compaia solo nella soluzione, scriveremo qualcosa del tipo:

```C
int pippo(int x)
{
#ifndef HANDOUT
   /* codice della soluzione che non sarà visibile nello "scheletro"
      da completare */
#else
   /* eventualmente, parte che sarà visibile SOLO nello scheletro
      da compilare e non nella soluzione */
#endif
}
```

Dando il comando `make` si generano in automatico i file HTML estratti
da ciascun esercizio (un file HTML per ogni esercizio), che vengono
copiati nella sottodirectory `handouts/`. Nelle sottodirectory
`handouts/` e `solutions/` vengono inoltre copiati i file sorgenti
eventualmente preprocessati come descritto sopra.

Il procedimento può essere schematizzato dal diagramma seguente:

```
+--------+ sed   +---------+ pandoc   +------------+
|        | ----> | file.md | -------> | file.html  |
|        |       +---------+          +------------+
|        |
|        | unifdef -DHANDOUT    +------------------+
| file.c | -------------------> | handouts/file.c  |
|        |                      +------------------+
|        |
|        | unifdef -UHANDOUT    +------------------+
|        | -------------------> | solutions/file.c |
+--------+                      +------------------+
```

