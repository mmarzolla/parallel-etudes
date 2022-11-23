#!/bin/sh

## Make a movie of the ANNEAL Callular Automaton

## Written by Moreno Marzolla on 2022-11-11
## Last updated 2022-11-22

DEFAULT_DEVICE=1 # simpleCL default device (0=usually the CPU)
NSTEPS=300000    # number of time steps
WIDTH=1920       # image resolution
HEIGHT=1080

if [ ! -f ./opencl-anneal-movie ]; then
    echo "FATAL: ./opencl-anneal-movie not found"
    exit 1
fi

## generate frames
SCL_DEFAULT_DEVICE=$DEFAULT_DEVICE ./opencl-anneal-movie $NSTEPS $WIDTH $HEIGHT

## insert annotations
for n in `seq 0 $NSTEPS`; do
    INPUTF=`printf "opencl-anneal-%06d.pbm" $n`
    OUTPUTF=`printf "opencl-anneal-%06d-out.png" $n`
    LABEL=`printf "%06d" $n`
    if [ -f "$INPUTF" ]; then
        echo "Annotating ${INPUTF}..."
        convert $INPUTF \
                -gravity southwest \
                -font Courier \
                -stroke black -strokewidth 15 -pointsize 100 -annotate 0 $LABEL \
                -stroke yellow -strokewidth 4 -fill yellow -annotate 0 $LABEL \
                $OUTPUTF
    fi
done

## make video @30fps
ffmpeg -pattern_type glob -y -i "opencl-anneal-*-out.png" -vcodec mpeg4 -r 30 opencl-anneal.avi
