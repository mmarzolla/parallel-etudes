#!/bin/bash

## Questo script effettua le seguenti sostituzioni da standard input a
## standard output:
##
## - #include "pgmutils.h" -> inserisce il contenuto del file pgmutils.h
##
## - #include "ppmutils.h" -> inserisce il contenuto del file ppmutils.h
##
## - \/\* -> \*
##
## - \*\/ -> */

sed -e '/#include "pgmutils.h"/ {' -e 'r pgmutils.h' -e 'd}' | \
    sed -e '/#include "ppmutils.h"/ {' -e 'r ppmutils.h' -e 'd}' | \
    sed s/\\\\\\/\\\\\\*/\/\\*/g | \
    sed s/\\\\\*\\\\\//\*\//g
