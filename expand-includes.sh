#!/bin/bash

## This script expands the following #includes:
##
## #include "pgmutils.h"
## #include "ppmutils.h"
##
## by copying the header files in place. Use this to make programs
## self-contained. The input file is read from stdin; the expanded
## file is written to stdout.

## Written by Moreno Marzolla on 2021-05-19
## Last updated on 2024-11-13 by Moreno Marzolla

cat "$1" | \
    sed -e '/\#include\s*"pgmutils.h"/ {' -e 'r pgmutils.h' -e 'd}' | \
    sed -e '/\#include\s*"ppmutils.h"/ {' -e 'r ppmutils.h' -e 'd}'
