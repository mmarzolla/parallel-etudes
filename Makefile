EXE_OMP := $(basename $(wildcard omp-*.c))
EXE_MPI := $(basename $(wildcard mpi-*.c))
EXE_SIMD := $(basename $(wildcard simd-*.c))
EXE_CUDA := $(basename $(wildcard cuda-*.cu)) cuda-anneal-shared cuda-nbody-shared
EXE_OPENCL := $(basename $(wildcard opencl-*.c)) opencl-anneal-local opencl-nbody-local opencl-anneal-movie
EXE_SERIAL := gensphfract genspheres gendna bbox-gen circles-gen knapsack-gen
EXE := $(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_SIMD) $(EXE_OPENCL) $(EXE_CUDA)
SRC := $(wildcard *.c) $(wildcard *.cu) $(wildcard *.cl)
INC := $(wildcard *.h)
DATAFILES := $(wildcard *.in) $(wildcard *.txt) $(wildcard *.gr) $(wildcard *.cnf) mandelbrot-set-demo.ggb
OUTFILES :=
HANDOUTS_SRC := ${SRC:%.c=handouts/%.c} ${SRC:%.cu=handouts/%.cu} ${SRC:%.cl=handouts/%.cl} ${INC:%.h=handouts/%.h}
SOLUTIONS_SRC := ${SRC:%.c=solutions/%.c} ${SRC:%.cu=solutions/%.cu} ${SRC:%.cl=solutions/%.cl} ${INC:%.h=solutions/%.h}
HTML := ${SRC:%.c=handouts/%.html} ${SRC:%.cu=handouts/%.html}
EXTRAS += lab.css $(wildcard *.png *.svg *.jpg *.pgm *.ppm *.md *.sh *.odp *.odg) mpi-rule30.pdf
IMGS := omp-c-ray-images.jpg denoise.png simd-map-levels.png edge-detect.png cat-map.png cat-map-demo.png anneal-demo.png
CFLAGS += -std=c99 -Wall -Wpedantic
LDLIBS +=
PANDOC_EXTRA_OPTS += -V lang=en-US --mathjax="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
NVCC ?= nvcc
MPICC ?= mpicc
NVCFLAGS += -Wno-deprecated-gpu-targets

.PHONY: MAKE_DIRS clean distclean images count-locs

ALL: MAKE_DIRS ${EXE} ${HTML} ${HANDOUTS_SRC} ${SOLUTIONS_SRC} ${OUTFILES}
	@cp -a -u ${EXTRAS} ${DATAFILES} ${OUTFILES} handouts/

images: ${IMGS}

MAKE_DIRS:
	@mkdir -p handouts solutions

$(EXE_OMP): CFLAGS+=-fopenmp
openmp: $(EXE_OMP)

$(EXE_MPI): CC=$(MPICC)
mpi: $(EXE_MPI)

$(EXE_SIMD): CFLAGS+=-march=native -O2 -g -ggdb
simd: $(EXE_SIMD)

$(EXE_OPENCL): CFLAGS+=simpleCL.c
$(EXE_OPENCL): LDLIBS+=-lOpenCL
opencl: $(EXE_OPENCL)

opencl-anneal: LDLIBS+=-lm

opencl-anneal-local: CFLAGS+=-DUSE_LOCAL
opencl-anneal-local: LDLIBS+=-lm
opencl-anneal-local: opencl-anneal.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

opencl-anneal-movie: CFLAGS+=-DDUMPALL
opencl-anneal-movie: LDLIBS+=-lm
opencl-anneal-movie: opencl-anneal.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

opencl-nbody: LDLIBS+=-lm

opencl-nbody-simd: LDLIBS+=-lm

opencl-coupled-oscillators: LDLIBS+=-lm

opencl-knapsack: LDLIBS+=-lm

opencl-nbody-local: CFLAGS+=-DUSE_LOCAL
opencl-nbody-local: LDLIBS+=-lm
opencl-nbody-local: opencl-nbody.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

omp-tri-gemv: LDLIBS+=-lm

cuda: $(EXE_CUDA)

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

cuda-anneal-shared: NVCFLAGS+=-DUSE_SHARED
cuda-anneal-shared: cuda-anneal.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

omp-c-ray: LDLIBS+=-lm

omp-cat-map: CFLAGS+=-O0

mpi-pi: LDLIBS+=-lm

omp-pi: LDLIBS+=-lm

omp-nbody: LDLIBS+=-lm

omp-knapsack: LDLIBS+=-lm

mpi-dot: LDLIBS+=-lm

mpi-bbox: LDLIBS+=-lm

mpi-nbody: LDLIBS+=-lm

mpi-c-ray: LDLIBS+=-lm

mpi-mandelbrot-area: LDLIBS+=-lm

gendna: LDLIBS+=-lm

omp-bellman-ford: LDLIBS+=-lm

cuda-nbody-shared: NVCFLAGS+=-DUSE_SHARED
cuda-nbody-shared: cuda-nbody.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

handouts/%.html: %.md
	pandoc -s $(PANDOC_EXTRA_OPTS) --from markdown --css lab.css --to html5 $< > $@

%.md: %.c
	./extract-markdown.sh $< > $@

%.md: %.cu
	./extract-markdown.sh $< > $@

%.pdf: %.md
	pandoc --from markdown $< -o $@

sphfract.small.in sphfract.big.in: gensphfract
	./gensphfract 4 > sphfract.small.in
	./gensphfract 6 > sphfract.big.in

spheres.in: genspheres
	./genspheres > $@

dna.in: gendna
	./gendna > $@

omp-c-ray-images.jpg: omp-c-ray sphfract.small.in spheres.in dna.in
	./omp-c-ray -r 10 < sphfract.small.in > c-ray1.tmp.ppm
	./omp-c-ray -r 10 < spheres.in > c-ray2.tmp.ppm
	./omp-c-ray -r 10 < dna.in > c-ray3.tmp.ppm
	montage c-ray?.tmp.ppm -tile 3x1 -geometry +2+4 $@
	\rm -f c-ray?.tmp.ppm

simd-map-levels.png: simd-map-levels simd-map-levels-in.pgm
	./simd-map-levels 100 180 < simd-map-levels-in.pgm > simd-map-levels.tmp.pgm
	montage simd-map-levels-in.pgm simd-map-levels.tmp.pgm -tile 2x1 -geometry +2+4 -resize 600x $@
	\rm -f simd-map-levels.tmp.pgm

edge-detect.png: omp-edge-detect BWstop-sign.pgm
	./omp-edge-detect < BWstop-sign.pgm > BWstop-sign-edges.pgm
	montage BWstop-sign.pgm BWstop-sign-edges.pgm -tile 2x1 -geometry +2+4 -resize 400x $@

cat-map.png: omp-cat-map
	for niter in 0 1 2 5 10 36; do \
	  ./omp-cat-map "$${niter}" < cat1368.pgm > `printf "cat-map-demo-%02d.pgm" $${niter}` ; \
	done
	montage "cat-map-demo-[0-9]*.pgm" -scale x300 -tile 6x1 -geometry +2+4 $@
	\rm -f "cat-map-demo-[0-9]*.pgm"

cat-map-demo.png: omp-cat-map
	for niter in 0 1 2 5 10 36; do \
	  ./omp-cat-map "$${niter}" < cat1368.pgm > "cat-map-demo-$${niter}.pgm" ; \
	  convert "cat-map-demo-$${niter}.pgm" -resize 128 -pointsize 18 -background white label:"K = $$niter" -gravity Center -append `printf "cat-map-demo-%02d.png" $${niter}` ; \
	done
	montage "cat-map-demo-[0-9]*.png" -tile 6x1 -geometry +5+5 $@
	\rm -f "cat-map-demo-[0-9]*.pgm" "cat-map-demo-[0-9]*.png"

anneal-demo.png: opencl-anneal
	for niter in 0 10 100 1000; do \
	  FILENAME=`printf "opencl-anneal-%06d.pbm" $${niter}` ; \
	  ./opencl-anneal "$${niter}" ; \
	  convert "$${FILENAME}" -resize 256 -pointsize 18 -background white label:"$$niter Iterations" -gravity Center -append `printf "anneal-demo-%06d.png" $${niter}` ; \
	done
	montage "anneal-demo-[0-9]*.png" -tile 4x1 -geometry +5+5 $@
	\rm -f "opencl-anneal-[0-9]*.pbm" "anneal-demo-[0-9]*.png"

valve-noise.ppm: valve.png
	convert $< -attenuate 0.2 +noise impulse -format ppm $@

denoise.png: omp-denoise valve-noise.ppm
	./omp-denoise < valve-noise.ppm > valve-clear.tmp.ppm
	montage valve-noise.ppm valve-clear.tmp.ppm -tile 2x1 -geometry +2+4 $@
	\rm -f valve-clear.tmp.ppm

handouts/%.c: %.c
	./expand-includes.sh $< | unifdef -x2 -DSERIAL > $@

handouts/%.cu: %.cu
	./expand-includes.sh $< | unifdef -x2 -DSERIAL > $@

handouts/%.h: %.h
	unifdef -x2 -DSERIAL $< > $@

handouts/%.cl: %.cl
	./expand-includes.sh $< | unifdef -x2 -DSERIAL > $@

solutions/%.c: %.c
	./expand-includes.sh $< | unifdef -x2 -USERIAL > $@

solutions/%.cu: %.cu
	./expand-includes.sh $< | unifdef -x2 -USERIAL > $@

solutions/%.h: %.h
	unifdef -x2 -USERIAL $< > $@

solutions/%.cl: %.cl
	./expand-includes.sh $< | unifdef -x2 -USERIAL > $@

pub: ALL
	rsync -av --delete-after handouts/ ~/public_html/teaching/high-performance-computing/2024-2025/handouts && \
	rsync -av --delete-after solutions/ ~/public_html/teaching/high-performance-computing/2024-2025/solutions && \
	put-aruba.sh

clean:
	\rm -r -f *.html a.out *.o *.s ${EXE} anneal-*.pbm coupled-oscillators.ppm anneal.avi cuda-anneal-*.pbm cuda-anneal.avi opencl-anneal-*.pbm opencl-anneal-*-out.png opencl-anneal.avi cuda-rule30.pbm opencl-rule30.pbm *.tmp.p[pbg]m sphfract.ppm opencl-mandelbrot.ppm cuda-mandelbrot.ppm cat-map-demo-[0-9]*.png cat-map-demo-[0-9]*.pgm anneal-demo-[0-9]*.p?? count-locs.tex

distclean: clean
	\rm -rf handouts/* solutions/*

count-locs.tex: count-locs.awk
	cloc --quiet --hide-rate --by-file --csv *.c *.cl *.cu | gawk -f count-locs.awk > count-locs.tex
