#!/bin/bash

## Estrae il blocco markdown delimitato da /*** ***/ dal sorgente
## passato sulla riga di comando. E' essenziale che le direttive /***
## e ***/ compaiano su una riga a se stante (cioè non devono essere
## presenti altri caratteri non blank né prima né dopo).
##
## Questo script inoltre sostituisce \/\* con /* e \*\/ con */ in modo
## che sia possibile inserire commenti nel markdown presente
## all'interno dei blocchi di commento al codice.

## Scritto da Moreno Marzolla nel 2021
## Ultima modifica il 2022-09-08

## Prima versione basata su sed; purtroppo ho dovuto eliminarla perché
## non funziona nel caso siano presenti più blocchi di commenti
## markdown che vanno estratti

# cat "$1" | \
#     sed -n '/^\s*\/\*\*\*\s*$/,${p;/^\s*\*\*\*\/\s*$/q}' | sed '1d;$d' | \
#     sed "s/\\\\\\/\\\\\\*/\\/\*/g" | \
#     sed "s/\\\\\\*\\\\\//\*\//g"

awk '\
BEGIN { inside = 0; } \
/^\s*\*\*\*\/\s*$/ { inside = 0; } \
inside == 1 { print $0; }\
/^\s*\/\*\*\*\s*$/ { inside = 1; } \
' "$1" | \
    sed "s/\\\\\\/\\\\\\*/\\/\*/g" | \
    sed "s/\\\\\\*\\\\\//\*\//g"
