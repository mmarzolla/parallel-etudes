#!/bin/bash

## Espande alcune direttive "include" del file sorgente passato sulla
## riga di comando.

## Moreno Marzolla 2021-05-19

cat "$1" | \
    sed -e '/\#include\s*"pgmutils.h"/ {' -e 'r pgmutils.h' -e 'd}' | \
    sed -e '/\#include\s*"ppmutils.h"/ {' -e 'r ppmutils.h' -e 'd}'
