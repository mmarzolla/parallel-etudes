#!/bin/bash

## Performs the following replacements on stdin; write result
## to stdout:
##
## - #include "pgmutils.h" -> replace with the content of file pgmutils.h
##
## - #include "ppmutils.h" -> replace with the content of file ppmutils.h
##
## - \/\* -> \*
##
## - \*\/ -> */

sed -e '/#include "pgmutils.h"/ {' -e 'r pgmutils.h' -e 'd}' | \
    sed -e '/#include "ppmutils.h"/ {' -e 'r ppmutils.h' -e 'd}' | \
    sed s/\\\\\\/\\\\\\*/\/\\*/g | \
    sed s/\\\\\*\\\\\//\*\//g
