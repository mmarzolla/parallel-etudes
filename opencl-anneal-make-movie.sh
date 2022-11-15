#!/bin/sh

## Crea un filmano montando un certo numero di frame
## dell'automa cellulare ANNEAL.

## Scritto da Moreno Marzolla il 2021-11-19
## Ultimo aggiornamento 2022-11-11

NFRAMES=1000000  # number of time steps
RES=1080         # image resolution

if [ ! -f ./opencl-anneal-movie ]; then
    echo "FATAL: ./opencl-anneal-movie not found"
    exit 1
fi

## generate the frames
./opencl-anneal-movie $NFRAMES $RES

## insert annotations
for n in `seq 0 $NFRAMES`; do
    INPUTF=`printf "opencl-anneal-%06d.pbm" $n`
    OUTPUTF=`printf "opencl-anneal-%06d-out.pbm" $n`
    LABEL=`printf "%05d" $n`
    if [ -f "$INPUTF" ]; then
        echo "Annotating ${INPUTF}..."
        convert $INPUTF \
                -gravity southwest \
                -font Courier \
                -stroke black -strokewidth 10 -pointsize 100 -annotate 0 $LABEL \
                -stroke none -fill white -annotate 0 $LABEL \
                $OUTPUTF
    fi
done

## crea il video
ffmpeg -y -i "opencl-anneal-%05d-out.pbm" -vcodec mpeg4 opencl-anneal.avi
