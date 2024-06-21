# Parallel Etudes

This repository contains the source code of exercises that are used in
the lab sessions for the High Performance Computing (HPC) course,
University of Bologna.

The lab sessions are organized as follows: each exercise includes a
specification document in the form of a HTML page, and a working
serial implementation of the program to be realized. The goal is to
parallelize the serial program using the technology that has been
previously introduced in the class (OpenMP, MPI, CUDA/OpenCL). At the
end of the lab session, the solution to all exercises is made
available.

Some exercises are quite simple, while others are slightly more
complex. However, the level of difficulty is quite low since students
are expected to fully solve at least one exercise during each lab
session.

There are a couple of interesting points that differentiate these
exercises from other parallel programming exercises, or parallel
benchmarks:

- They require little or no knowledge outside parallel programming to
  be solved; in particular, they require very little or no knowledge
  of physics, linear algebra or numerical analysis.

- Many exercises are small applications that are designed to be
  interesting. These applications are taken from different domains,
  e.g., 3D graphics, Cellular Automata, gravitational N-body solvers,
  cryptography and so on. Some applications produce images or movies
  as output, rather than numbers or dry output messages.

- Some programs are parallelized using multiple programming paradigms.
  For example, the same N-body solver might be proposed during OpenMP
  lab sessions, then MPI, and then CUDA/OpenCL. This is quite useful
  to see how each parallel programming paradigm can be applied to the
  same problem.

## List of exercises

Table 1 lists, for each exercise, which parallel versions are
available, and which parallel programming patterns are used to solve
it.

Table 1: List of exercises

Kernel                           OpenMP   MPI   OpenCL  Pattern
------------------------------ --------- ----- -------- --------------------------
Password cracking                X         X            Embarrassingly parallel
Dot product                      X         X     X      Reduction, Scatter/Gather
Array sum                                  X            Reduction, Scatter/Gather
Monte Carlo Pi                   X         X            Embarrassingly parallel, Reduction
Sieve of Eratosthenes            X               X      Embarrassingly parallel, Reduction
Character frequencies            X         X     X      Embarrassingly parallel, Reduction
Inclusive scan                   X         X            Scan
Dynamic task scheduling          X                      Master-Worker
MergeSort                        X                      Task-level parallelism
Binary Tree traversal            X                      Task-level parallelism
Ray tracing                      X         X            Embarrassingly parallel, Scatter/Gather
Levenstein's distance            X                      2D stencil, wavefront
Ray casting                      X               X      Embarrassingly parallel
Arnold's cat map                 X               X      Embarrassingly parallel
Mandelbrot set                   X         X     X      Embarrassingly parallel, Load balancing
Area of the Mandelbrot set       X         X     X      Embarrassingly parallel, Load balancing, Reduction
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
SAT solver                       X                      Embarrassingly parallel
Single-Source Shortest Path      X                      Reduction
All-Pairs Shortest Paths         X                      Embarrassingly parallel
------------------------------ --------- ----- -------- --------------------------

The final exam of the HPC course requires the parallelization of a
slightly more complex application, which is provided by the instructor
in serial form. Over the years, the following programs (not available
in this repository) have been proposed as exercises:

Academic Year      Topic
-----------------  ----------------------------------------
2016/2017          Biham-Middleton-Levine CA
2017/2018          Larger-than-Life CA
2018/2019          Earthquake
2019/2020          Convex Hull
2020/2021          Skyline
2021/2022          Hardy, Pomeau, de Pazzis CA (HPP)
2022/2023          Smoothed Particle Hydrodynamics (SPH)
2023/2024          Force-directed circles drawing
-----------------  ----------------------------------------

## OpenMP

### Lab 1

- [Brute force password cracking](omp-brute-force.html) (embarrassingly parallel)
- [Dot product](omp-dot.html) (reduction)
- [Monte Carlo approximation of $\pi$](omp-pi.html) (reduction)
- [Character counts](omp-letters.html) (reduction)
- [Sieve of Eratosthenes](omp-sieve.html) (embarrassingly parallel)

### Lab 2

- [Dynamic scheduling](omp-schedule.html) (master-worker)
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
- [SAT Solver](omp-sat-solver.html) (embarrassingly parallel)
- [Single-Source Shortest Paths](omp-bellman-ford.html) (reduction)
- [Binary Tree traversal](omp-bintree-walk.html) (task)

## MPI

### Lab 1

- [Ring communication](mpi-ring.html) (point-to-point)
- [MPI_Bcase using tree-structured communications](mpi-my-bcast.html) (broadcast, point-to-point)
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
- [Sum-reduction of arbitrary array with point-to-point communications](mpi-sum.html) (reduction, point-to-point)
- [MPI_Reduce using tree-structured communications](mpi-reduce.html) (reduction, point-to-point)
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


## Prerequisites

To build the executables and documentation for the programs, the
following tools are required:

- [Pandoc](https://pandoc.org/)

- [Sed](https://www.gnu.org/software/sed/)

- [GNU make](https://www.gnu.org/software/make/)

- [unifdef](https://dotat.at/prog/unifdef/)

## Use

Type

    make

to generate the specification of each exercise in HTML format,
together with the source code of the skeleton provided during the lab
sessions and the corresponding solutions.

## How it works

The repository contains source files (with extensions `.c`, `.cu`,
`.cl` and `.h`) and data files. The specification of each exercise is
included in comment blocks in each source file; specifically, the
content of comments included within these markers:

```C
/***

...

***/
```

is treated as a Markdown
text. [Markdown](https://www.markdownguide.org/) is a text-based
markup language that allows formatting and structural elements to be
described with a minimalistic and unobstrusive syntax. The provided
`Makefile` extracts the content of the comments using `sed`, and
formats it using [pandoc](https://pandoc.org/index.html) to produce
HTML pages.

Each source files is further processed to produce a skeleton that is
provided to students during the lab sessions, and the complete
solution that is made available afterwards. To define which portion of
the source code goes to the skeleton or solution, it is possible to
use the `HANDOUT` preprocessor symbol: this symbol is defined when
compiling the skeleton, and is not defined when compiling the
solution. 

```C
int foo(int x)
{
#ifdef HANDOUT
   /* This block will be included in the skeleton provided
      to students. */
#else
   /* This block will be included in the solution */
#endif
}
```

The Makefile uses the [unifdef](https://dotat.at/prog/unifdef/)
program to generate new source files for both cases.

Therefore, from each source file (`.c` or `.cu`) the provided
Makefile generates:

- The specification of the assignment, by extracting the comments
  formatted as above and converting them to HTML and
  placed into the `handouts/` subdirectory;

- The source code that will be provided during the lab sessions as
  skeleton to be completed by the students, again placed into the
  `handouts/` subdirectory; all other source files (`.h` and `.cl`),
  plus any additional data file is also copied there.

- The source code of the solution, placed into the `solutions/`
  subdirectory.

The following figure illustrates the process:

```
+--------+ sed   +---------+ pandoc   +------------+
|        | ----> | file.md | -------> | file.html  |
|        |       +---------+          +------------+
|        |
|        | unifdef -DHANDOUT    +------------------+
| file.c | -------------------> | handouts/file.c  |
|        |                      +------------------+
|        |
|        | unifdef -UHANDOUT    +------------------+
|        | -------------------> | solutions/file.c |
+--------+                      +------------------+
```

