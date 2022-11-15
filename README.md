# LabASD

Questo repository contiene materiale usato per il corso di
Laboratorio di Algoritmi e Strutture Dati (LabASD), corso di studio
in Ingegneria e Scienze Informatiche, Università di Bologna.

## Strumenti

Per elaborare i sorgenti contenuti sono necessari le seguenti
applicazioni:

- [Pandoc](https://pandoc.org/)

- [Sed](https://www.gnu.org/software/sed/)

- [GNU make](https://www.gnu.org/software/make/)

- [unifdef](https://dotat.at/prog/unifdef/)

- [splint](https://splint.org/) opzionale, serve per l'analisi
  statica dei sorgenti

## Come funziona

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

Nel Makefile sono definiti i seguenti target:

`ALL`

: (default) compila tutti i file sorgente, genera gli scheletri
  dei programmi e le soluzioni, genera il testo delle esercitazioni
  in formato HTML, e copia le immagini e altri file di supporto
  nelle directory `handouts/` e `solutions/`.

`dist`

: produce un archivio in formato `.tar.gz` che include tutti i file

`pub`

: copia il contenuto delle directory `handouts/` e `solutions/`
  nella copia locale del mio sito Web personale

`clean`

: rimuove i file temporanei e gli eseguibili

`distclean`

: come `clean`, ma in più rimuove tutto il contenuto delle directory
  `handouts/` e `solutions/`

`lint`

: analizza in modo statico tutti i sorgenti utilizzando
  [splint](https://splint.org/). L'analisi statica aiuta a trovare
  potenziali errori o ambiguità nel codice.

Tutti i programmi sono stati compilati senza errori né _warning_
usando [GCC](https://gcc.gnu.org/), [clang](https://clang.llvm.org/) e
[tcc](https://bellard.org/tcc/). I parametri passati sulla riga di
comando vengono riconosciuti da tutti questi compilatori.  Per
compilare, ad es., con `ŧcc` si può usare la riga di comando:

    CC=tcc make
