# Parallel Etudes

_Etudes for Programmers_ is an unusual yet influential book written by
Charles Wetherell and published in 1978. The term _étude_ refers to a
small musical piece that is intended for learning to play an
instrument.  The book argues that programming is a craft that -- at
least in part -- is learned by practicing, like playing a musical
instrument. To this aim, the book proposes a set of programming
exercises of various levels of difficulty, from extremely simple to
very complex.

I strongly agree with Wetherell, and believe that the practice-based
approach he suggests is very appropriate also for learning _parallel_
programming. I incorporated this idea in the [High Performance
Computing](https://www.moreno.marzolla.name/teaching/HPC/) course that
I have been teaching over the past 10 years to Computer Science and
Engineering students at the University of Bologna.

The course is an elective, undergraduate course in parallel
programming; it covers all the major parallel programming models,
i.e., shared-memory, distributed-memory and GPU. Considerable emphasis
is put on practical aspects of parallel programming using OpenMP, MPI
and CUDA, for which lab exercises need to be developed.

This repository contains the source code of programming exercises that
are used in the lab sessions. The labs are organized as follows: each
exercise includes a detailed specification, and is provided with a
fully functional serial solution. The goal is to parallelize the
serial program using the techniques discussed in the previous
classes. Reference parallel solutions are made available at the end of
each lab session, so that each student can compare his/her own code to
the program provided by the instructor.

In the spirit of Wetherell's _etudes_, some exercises are simple while
others are more complex. However, the overall level of difficulty is
moderate, since students are expected to solve at least one exercise
during each lab session.

Some notable points:

- These exercises require little or no knowledge outside parallel
  programming; in particular, they do not require advanced knowledge
  of physics, linear algebra or numerical analysis.

- Many exercises are designed to be interesting, and are taken from
  different domains such as 3D graphics, Cellular Automata,
  gravitational N-body simulations, cryptography, and others. Some
  programs generate images or movies, to make them more appealing.

- Some exercises can be parallelized using multiple programming
  paradigms. This is quite useful to appreciate the strengths and
  weaknesses of each paradigm.

## Citation

The teaching methodology behind this collection of parallel
programming exercises has been described in the following paper:

> Moreno Marzolla, _Etudes for Parallel Programmers_, proc. 33rd
> Euromicro International Conference on Parallel, Distributed, and
> Network-Based Processing (PDP), march 12--14 2025, Turin, Italy,
> pp. 341--348, IEEE Computer Society Conference Publishing Services
> (CPS) ISBN: 979-8-3315-2493-7 ISSN: 2377-5750, doi:
> <https://doi.org/10.1109/PDP66500.2025.00010>

BibTeX:

```
@inproceedings{parallel-etudes,
  author = "Moreno Marzolla",
  title = "Etudes for Parallel Programmers",
  booktitle = "proc. 33rd Euromicro International Conference on Parallel, Distributed, and Network-Based Processing (PDP)",
  editor = "Alessia Antelmi and Iacopo Colonnelli and Doriana Medi{\'{c}} and Horacio Gonz{\'{a}}lez-V{\'{e}}lez",
  month = mar # "12--14",
  address = "Turin, Italy",
  year = 2025,
  pages = "341--348",
  isbn = "979-8-3315-2493-7",
  organization = "Euromicro",
  publisher = "IEEE CPS",
  doi = "10.1109/PDP66500.2025.00010"
}
```

## List of exercises

Table 1 lists, for each exercise, which parallel versions are
available, and which parallel programming patterns are used to solve
it.

: Table 1: List of exercises

| Kernel                      | OpenMP | MPI | CUDA | OpenCL | Pattern                                            |
|-----------------------------|--------|-----|------|--------|----------------------------------------------------|
| Password cracking           | X      | X   |      |        | Embarrassingly parallel                            |
| Dot product                 | X      | X   | X    | X      | Reduction, Scatter/Gather                          |
| Circular shift of array     | X      | X   |      |        | Point-to-point                                     |
| Array sum                   |        | X   |      |        | Reduction, Scatter/Gather                          |
| Monte Carlo Pi              | X      | X   |      |        | Embarrassingly parallel, Reduction                 |
| Sieve of Eratosthenes       | X      |     | X    | X      | Embarrassingly parallel, Reduction                 |
| Character frequencies       | X      | X   | X    | X      | Embarrassingly parallel, Reduction                 |
| Inclusive scan              | X      | X   |      |        | Scan                                               |
| OpenMP `schedule()`         | X      | NA  | NA   | NA     | OpenMP loop scheduling                             |
| Image erosion               | X      | NA  | NA   | NA     | 2D Stentil, OpenMP loop collapse                   |
| MergeSort                   | X      | NA  | NA   | NA     | Task-level parallelism                             |
| Binary Tree traversal       | X      | NA  | NA   | NA     | Task-level parallelism                             |
| Ray tracing                 | X      | X   |      |        | Embarrassingly parallel, Scatter/Gather            |
| Levenstein's distance       | X      |     |      |        | 2D stencil, wavefront                              |
| Arnold's cat map            | X      |     | X    | X      | Embarrassingly parallel                            |
| Mandelbrot set              | X      | X   | X    | X      | Embarrassingly parallel, Load balancing            |
| Area of the Mandelbrot set  | X      | X   | X    | X      | Embarrassingly parallel, Load balancing, Reduction |
| Image denoising             | X      |     | X    | X      | 2D Stencil                                         |
| List ranking                | X      | NA  |      |        | Pointer Jumping                                    |
| Area of union of circles    |        | X   |      |        | Embarrassingly parallel, Scatter/Gather, Reduction |
| Bounding Box                |        | X   |      |        | Scatter/Gather, Reduction                          |
| Rule 30 CA                  | X      | X   | X    | X      | 1D Stencil, Point-to-point                         |
| Linear search               |        | X   |      |        | Embarrassingly parallel, Reduction                 |
| Binary search               | X      |     |      | X      | Divide-and-conquer                                 |
| Odd-Even Sort               | X      | X   | X    | X      | Scatter/Gather, Point-to-point                     |
| Coupled oscillators         |        |     | X    | X      | 1D Stencil                                         |
| Anneal CA                   |        |     | X    | X      | 2D Stencil                                         |
| N-body simulation           | X      |     | X    | X      | Embarrassingly parallel, Load balancing, Reduction |
| Knapsack problem            | X      |     | X    | X      | Non-uniform 1D stencil                             |
| Edge detection              | X      |     | X    | X      | 2D Stencil                                         |
| Gaussian elimination        | X      |     |      |        | Reduction                                          |
| SAT solver                  | X      | X   | X    | X      | Embarrassingly parallel, Reduction                 |
| Single-Source Shortest Path | X      |     |      |        | Reduction                                          |
| All-Pairs Shortest Paths    | X      |     | X    | X      | (Almost) Embarrassingly parallel                   |

## Prerequisites

To build the executables and documentation for the programs, the
following tools are required:

- [Pandoc](https://pandoc.org/)

- [Sed](https://www.gnu.org/software/sed/)

- [GNU make](https://www.gnu.org/software/make/)

- [unifdef](https://dotat.at/prog/unifdef/)

## Using these exercises

Type

    make

to generate the specification of each exercise in HTML format,
together with the skeleton source code provided during the lab
sessions, and the corresponding solution.

## How it works

The repository contains program sources (with extensions `.c`, `.cu`,
`.cl` and `.h`) and data files. The specification of each exercise is
in a comment blocks at the top of the source file; specifically, all
text between the markers

```C
/***

...

***/
```

is treated as Markdown-formatted specification
text. [Markdown](https://www.markdownguide.org/) is a text-based
markup language that allows formatting and structural elements to be
described with a simple syntax. The provided `Makefile` extracts the
content of the comment blocks above and formats it using
[pandoc](https://pandoc.org/index.html) to produce HTML pages.

Each source file is further processed to produce both the serial
program provided to students during the lab sessions, and the solution
that is made available afterwards. To define which portion of the
source code goes to the serial program or the solution, we use the
`SERIAL` preprocessor symbol: this symbol is defined when compiling
the serial code only.

```C
int foo(int x)
{
#ifdef SERIAL
   /* This will appear in the serial program. */
#else
   /* This will appear in the solution. */
#endif
   /* This will appear in both the serial program
      an d the solution. */
}
```

The Makefile uses the [unifdef](https://dotat.at/prog/unifdef/)
utility to generate new source files for both cases.

Therefore, from each source file (`.c` or `.cu`) the Makefile
generates:

- The specification of the assignment, by extracting the comments
  formatted as above and converting them to HTML placed inside the
  `handouts/` subdirectory;

- The serial code provided during the lab sessions, to be used as the
  starting point for the parallel version, placed into the `handouts/`
  subdirectory; all other source files (`.h` and `.cl`), plus any
  additional data file, are also copied there.

- The source code of the parallel solution, placed into the
  `solutions/` subdirectory.

The following figure illustrates the process:

```
+--------+ sed   +---------+ pandoc   +------------+
|        | ----> | file.md | -------> | file.html  |
|        |       +---------+          +------------+
|        |
|        | unifdef -DSERIAL     +------------------+
| file.c | -------------------> | handouts/file.c  |
|        |                      +------------------+
|        |
|        | unifdef -USERIAL     +------------------+
|        | -------------------> | solutions/file.c |
+--------+                      +------------------+
```

