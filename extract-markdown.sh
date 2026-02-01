#!/usr/bin/env bash

## Use this script to extract comment blocks delimited by `/***` and
## `***/`. The delimiters *must* appear on a line from themselves,
## without any extraneous characters before or after (spaces are ok).
##
## This script also replaces `\/\*` with `/*`, and `\*\/` with `*/` so
## that it is possible to type C-style comments inside comment blocks.

## Usage: ./extract-markdown.sh filename > out

## Written by Moreno Marzolla in 2021
## Last updated on 2024-11-13 by Moreno Marzolla

awk '\
BEGIN { inside = 0; } \
/^\s*\*\*\*\/\s*$/ { inside = 0; } \
inside == 1 { print $0; }\
/^\s*\/\*\*\*\s*$/ { inside = 1; } \
' "$1" | \
    sed "s/\\\\\\/\\\\\\*/\\/\*/g" | \
    sed "s/\\\\\\*\\\\\//\*\//g"
