EXE_OMP := $(basename $(wildcard omp-*.c))
EXE_MPI := $(basename $(wildcard mpi-*.c))
EXE_SIMD := $(basename $(wildcard simd-*.c))
EXE_CUDA := $(basename $(wildcard cuda-*.cu)) cuda-anneal-shared cuda-nbody-shared
EXE_OPENCL := $(basename $(wildcard opencl-*.c)) opencl-anneal-local opencl-nbody-local opencl-anneal-movie
EXE_SERIAL := gen-sphfract gen-spheres gen-dna gen-bbox gen-circles gen-knapsack gen-graph
EXE := $(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_SIMD) $(EXE_OPENCL) $(EXE_CUDA)
SRC := $(wildcard *.c) $(wildcard *.cu) $(wildcard *.cl)
INC := $(wildcard *.h)
DATAFILES := $(wildcard *.in) $(wildcard *.txt) $(wildcard *.gr) $(wildcard *.cnf) mandelbrot-set-demo.ggb
OUTFILES :=
HANDOUTS_SRC := ${SRC:%.c=handouts/%.c} ${SRC:%.cu=handouts/%.cu} ${SRC:%.cl=handouts/%.cl} ${INC:%.h=handouts/%.h}
SOLUTIONS_SRC := ${SRC:%.c=solutions/%.c} ${SRC:%.cu=solutions/%.cu} ${SRC:%.cl=solutions/%.cl} ${INC:%.h=solutions/%.h}
HTML := ${SRC:%.c=handouts/%.html} ${SRC:%.cu=handouts/%.html}
EXTRAS += parallel-etudes.css $(wildcard *.png *.svg *.jpg *.pgm *.ppm *.md *.sh *.odp *.odg) mpi-rule30.pdf
IMGS := omp-c-ray-images.jpg denoise.png simd-map-levels.png edge-detect.png cat-map.png cat-map-demo.png anneal-demo.png parallel-etudes.jpg
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

mpi-gemv: LDLIBS+=-lm

omp-pi: LDLIBS+=-lm

omp-nbody: LDLIBS+=-lm

omp-knapsack: LDLIBS+=-lm

mpi-dot: LDLIBS+=-lm

mpi-bbox: LDLIBS+=-lm

mpi-nbody: LDLIBS+=-lm

mpi-c-ray: LDLIBS+=-lm

mpi-mandelbrot-area: LDLIBS+=-lm

gen-dna: LDLIBS+=-lm

omp-bellman-ford: LDLIBS+=-lm

cuda-nbody-shared: NVCFLAGS+=-DUSE_SHARED
cuda-nbody-shared: cuda-nbody.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

handouts/%.html: %.md
	pandoc -s $(PANDOC_EXTRA_OPTS) --from markdown --css parallel-etudes.css --to html5 $< > $@

%.md: %.c
	./extract-markdown.sh $< > $@

%.md: %.cu
	./extract-markdown.sh $< > $@

%.pdf: %.md
	pandoc --from markdown $< -o $@

sphfract.small.in sphfract.big.in: gen-sphfract
	./gen-sphfract 4 > sphfract.small.in
	./gen-sphfract 6 > sphfract.big.in

spheres.in: gen-spheres
	./gen-spheres > $@

dna.in: gen-dna
	./gen-dna > $@

sphfract.small.tmp.ppm: omp-c-ray sphfract.small.in
	./omp-c-ray -r 10 < sphfract.small.in > $@

spheres.tmp.ppm: omp-c-ray spheres.in
	./omp-c-ray -r 10 < spheres.in > $@

dna.tmp.ppm: omp-c-ray dna.in
	./omp-c-ray -r 10 < dna.in > $@

omp-c-ray-images.jpg: sphfract.small.tmp.ppm spheres.tmp.ppm dna.tmp.ppm
	montage $< -tile 3x1 -geometry +2+4 $@

simd-map-levels.png: simd-map-levels simd-map-levels-in.pgm
	./simd-map-levels 100 180 < simd-map-levels-in.pgm > simd-map-levels.tmp.pgm
	montage simd-map-levels-in.pgm simd-map-levels.tmp.pgm -tile 2x1 -geometry +2+4 -resize 600x $@
	\rm -f simd-map-levels.tmp.pgm

edge-detect.png: omp-edge-detect BWstop-sign.pgm
	./omp-edge-detect < BWstop-sign.pgm > BWstop-sign-edges.pgm
	montage BWstop-sign.pgm BWstop-sign-edges.pgm -tile 2x1 -geometry +2+4 -resize 400x $@

cat-map.png: omp-cat-map
	for niter in 0 1 2 5 10 36; do \
	  ./omp-cat-map "$${niter}" < cat1368.pgm > `printf "cat-map-demo-%02d.tmp.pgm" $${niter}` ; \
	done
	montage "cat-map-demo-[0-9]*.tmp.pgm" -scale x300 -tile 6x1 -geometry +2+4 $@
	\rm -f "cat-map-demo-[0-9]*.tmp.pgm"

cat-map-demo.png: omp-cat-map
	for niter in 0 1 2 5 10 36; do \
	  ./omp-cat-map "$${niter}" < cat1368.pgm > "cat-map-demo-$${niter}.tmp.pgm" ; \
	  convert "cat-map-demo-$${niter}.tmp.pgm" -resize 128 -pointsize 18 -background white label:"K = $$niter" -gravity Center -append `printf "cat-map-demo-%02d.tmp.png" $${niter}` ; \
	done
	montage "cat-map-demo-[0-9]*.tmp.png" -tile 6x1 -geometry +5+5 $@
	\rm -f "cat-map-demo-[0-9]*.tmp.pgm" "cat-map-demo-[0-9]*.tmp.png"

anneal-demo.png: opencl-anneal
	for niter in 0 10 100 1000; do \
	  FILENAME=`printf "opencl-anneal-%06d.pbm" $${niter}` ; \
	  ./opencl-anneal "$${niter}" ; \
	  convert "$${FILENAME}" -resize 256 -pointsize 18 -background white label:"$$niter Iterations" -gravity Center -append `printf "anneal-demo-%06d.tmp.png" $${niter}` ; \
	done
	montage "anneal-demo-[0-9]*.tmp.png" -tile 4x1 -geometry +5+5 $@
	\rm -f "opencl-anneal-[0-9]*.pbm" "anneal-demo-[0-9]*.tmp.png"

valve-noise.tmp.ppm: valve.png
	convert $< -attenuate 0.2 +noise impulse -format ppm $@

valve-clear.tmp.ppm: omp-denoise valve-noise.tmp.ppm
	./omp-denoise < valve-noise.tmp.ppm > $@

denoise.png: valve-noise.tmp.ppm valve-clear.tmp.ppm
	montage $< -tile 2x1 -geometry +2+4 $@

# cover image
parallel-etudes.jpg: dna.tmp.ppm omp-anneal mandelbrot-set.png rule30.png
	./omp-anneal 200
	montage -scale x480 mandelbrot-set.png omp-anneal-000200.pbm dna.tmp.ppm rule30.png -tile 4x1 -geometry +2+4 $@
	\rm -f omp-anneal-000200.pbm

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

pub: MAKE_DIRS ${HTML} ${HANDOUTS_SRC} ${SOLUTIONS_SRC} ${OUTFILES}
	@cp -a -u ${EXTRAS} ${DATAFILES} ${OUTFILES} handouts/
	rsync -av --delete-after handouts/ ~/public_html/teaching/calcolo-parallelo/2025-2026/handouts && \
	rsync -av --delete-after solutions/ ~/public_html/teaching/calcolo-parallelo/2025-2026/solutions && \
	put-aruba

clean:
	\rm -r -f *.html a.out *.o *.s ${EXE} *-anneal.avi coupled-oscillators.ppm opencl-anneal-*-out.png cuda-rule30.pbm opencl-rule30.pbm *.tmp.p?? *-mandelbrot.ppm count-locs.tex

distclean: clean
	\rm -rf handouts/* solutions/*

count-locs:
	cloc --quiet --hide-rate --by-file --csv *.c *.cl *.cu | gawk -f count-locs.awk > count-locs.tex
