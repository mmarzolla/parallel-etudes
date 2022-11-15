#!/bin/sh

## Plot the recurrence time of images of any size between 1 and 1024.
## This script requires omp-cat-map-rectime compiled with OpenMP
## (the serial version takes way too long), and gnuplot.

## Written by Moreno Marzolla on 2022-08-12
## Last modified by Moreno Marzolla on 2022-08-12

if [ ! -f ./omp-cat-map-rectime ]; then
    echo "FATAL: ./omp-cat-map-rectime is missing."
    exit 1
fi

echo "# N rec_time" > cat-map-rectime.txt
for n in `seq 1 1024`; do
         echo -n "$n "
	./omp-cat-map-rectime $n | head -1
done >> cat-map-rectime.txt
gnuplot <<EOF
set term png notransparent linewidth 2
set output "cat-map-rectime.png"
set xlabel "Image size N"
set ylabel "Minimum recurrence time"
set autoscale fix
plot [][0:] "cat-map-rectime.txt" with p notitle
EOF
