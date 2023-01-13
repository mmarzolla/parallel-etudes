% HPC - List of exercises
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-03


Kernel                           OpenMP   MPI   OpenCL  Pattern
------------------------------ --------- ----- -------- --------------------------
Password cracking                X                      Embarrassingly parallel
Dot product                      X         X     X      Reduction, Scatter/Gather
Array sum                                  X            Reduction, Scatter/Gather
Monte Carlo Pi                   X         X            Embarrassingly parallel, Reduction
Primes                           X                      Embarrassingly parallel, Reduction
Character frequencies            X                      Embarrassingly parallel, Reduction
Inclusive scan                   X                      Scan
Dynamic task scheduling          X                      Master-Worker
MergeSort                        X                      Task-level parallelism
Ray tracing                      X         X            Embarrassingly parallel, Scatter/Gather
Levenstein's distance            X                      2D stencil, wavefront
Ray casting                      X               X      Embarrassingly parallel
Arnold's cat map                 X               X      Embarrassingly parallel
Mandelbrot set                   X         X     X      Embarrassingly parallel, Load balancing
Area of the Mandelbrot set       X                      Embarrassingly parallel, Load balancing, Reduction
Image denoising                  X               X      2D Stencil
List ranking                     X                      Pointer Jumping
Area of union of circles                   X            Embarrassingly parallel, Scatter/Gather, Reduction
Bounding Box                               X            Scatter/Gather, Reduction
Rule 30 CA                       X         X     X      1D Stencil, Point-to-point
Linear search                              X            Embarrassingly parallel, Reduction
Binary search                    X               X      Divide-and-conquer
Odd-Even Sort                    X         X     X      Scatter/Gather, Point-to-point
Coupled oscillators                              X      1D Stencil
Anneal CA                                        X      2D Stencil
N-body simulation                X               X      Embarrassingly parallel, Load balancing, Reduction
Knapsack problem                 X               X      Non-uniform 1D stencil
Edge detection                   X               X      2D Stencil
Gaussian elimination             X                      Reduction
------------------------------ --------- ----- -------- --------------------------

## OpenMP

### Lab 1

- [Brute force password cracking](omp-brute-force.html) (embarrassingly parallel)
- [Dot product](omp-dot.html) (reduction)
- [Monte Carlo approximation of $\pi$](omp-pi.html) (reduction)
- [Character counts](omp-letters.html) (reduction)
- [Sieve of Eratosthenes](omp-sieve.html) (embarrassingly parallel)

### Lab 2

- [Dynamic scheduling](omp-dynamic.html) (master-worker)
- [Loop-carried dependencies](omp-loop.html)
- [MergeSort](omp-mergesort.html) (task-level parallelism)
- [Ray tracing](omp-c-ray.html) (embarrassingly parallel)
- [Arnold's Cat Map](omp-cat-map.html) (embarrassingly parallel)

### Extra

- [Inclusive scan](omp-inclusive-scan.html) (scan)
- [Area of the Mandelbrot set](omp-mandelbrot-area.html) (reduction)
- [Levenshtein edit distance](omp-levenshtein.html) (2D stencil, wavefront)
- [Image denoising](omp-denoise.html) (2D stencil)
- [Edge detect](omp-edge-detect.html) (2D stencil)
- [List ranking](omp-list-ranking.html) (pointer jumping)
- [Static and dynamic loop scheduling](omp-schedule.html) (master-worker)
- [N-body simulation](omp-nbody.html) (embarrassingly parallel, load balancing, reduction)
- [0/1 knapsack problem](omp-knapsack.html) (2D stencil)
- [Gaussian elimination](omp-gaussian-elimination.html) (reduction)

## MPI

### Lab 1

- [Ring communication](mpi-ring.html) (point-to-point)
- [Simulating broadcast](mpi-my-bcast.html) (point-to-point)
- [Monte Carlo approximation of $\pi$](mpi-pi.html) (reduction)
- [Sum-reduction](mpi-sum.html) (reduction)

### Lab 2

- [Mandelbrot set](mpi-mandelbrot.html) (embarrassingly parallel, scatter/gather)
- [Dot product](mpi-dot.html) (reduction, scatter/gather)
- [Area of the union of circles](mpi-circles.html) (embarrassingly parallel, reduction)

### Lab 3

- [Bounding Box](mpi-bbox.html) (scatter/gather, reduction)
- [MPI Datatype](mpi-send-col.html)
- [Rule 30 CA](mpi-rule30.html) (1D stencil, point-to-point)
- [Parallel search](mpi-lookup.html)

## Extra

- [Scatter](mpi-my-scatter.html)
- [Sum-reduction with point-to-point communications](mpi-sum.html) (reduction, point-to-point)
- [Circular shif of arrayt](mpi-rotate-right.html) Esame 2022-01-12
- [First occurrence](mpi-first-pos.html) Esame 2022-02-09
- [Ray Tracer](mpi-c-ray.html) (embarrassingly parallel, scatter/gather)

## CUDA

### Lab 1

- [Array inversion](cuda-reverse.html)
- [Dot product](cuda-dot.html) (reduction)
- [Odd-Even transposition sort](cuda-odd-even.html)
- [Coupled oscillators](cuda-coupled-oscillators.html) (1D stencil)

### Lab 2

- [Dense matrix sum](cuda-matsum.html)
- [Rule 30 CA](cuda-rule30.html) (1D stencil)
- [ANNEAL CA](cuda-anneal.html) (2D stencil)
- [Arnold's cat map](cuda-cat-map.html) (embarrassingly parallel)

### Extra

- [Image denoising](cuda-denoise.html) (2D stencil)
- [N-body simulation](cuda-nbody.html) (embarrassingly parallel, reduction)
- [0-1 knapsack problem](cuda-knapsack.html) (nonuniform 1D stencil)

## OpenCL

### Lab 1

- [Array inversion](opencl-reverse.html)
- [Dot product](opencl-dot.html)
- [Odd-Even transposition sort](opencl-odd-even.html)
- [Coupled oscillators](opencl-coupled-oscillators.html)

### Lab 2

- [Dense matrix sum](opencl-matsum.html)
- [Rule 30 CA](opencl-rule30.html)
- [ANNEAL CA](opencl-anneal.html)
- [Arnold's cat map](opencl-cat-map.html)

### Extra

- [Image denoising](opencl-denoise.html)
- [Edge detect](opencl-edge-detect.html) (2D stencil)
- [N-body simulation](opencl-nbody.html)
- [0-1 knapsack problem](opencl-knapsack.html)
- [Mandelbrot set](opencl-mandelbrot.html)
- [Binary Search](opencl-bsearch.html)

## SIMD

### Lab 1

- [Dot product](simd-dot.html)
- [Matirx-vector multiply](simd-matmul.html)
- [Gray level mapping](simd-map-levels.html)
- [Arnold's cat map](simd-cat-map.html)

## Final projects

Year               Topic
-----------------  ---------------------------
2016/2017          Biham-Middleton-Levine CA
2017/2018          Larger-than-Life
2018/2019          Earthquake
2019/2020          Convex Hull
2020/2021          Skyline
2021/2022          Hardy, Pomeau, de Pazzis CA
2022/2023          ???
-----------------  ---------------------------

