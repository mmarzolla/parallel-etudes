# Parallel Etudes

_Etudes for Programmers_ is an unusual yet influential book written by
Charles Wetherell and published in 1978. The term _étude_ refers to a
small musical piece that is intended for learning to play an
instrument.  The book argues that, like playing a musical instrument,
programming is a craft that -- at least in part -- is learned by
practicing. To this aim, the book proposes a set of programming
exercises of various levels of difficulty, from extremely simple to
extremely complex.

I strongly agree with Wetherell, and believe that the practice-based
approach he suggests is very appropriate also for learning _parallel_
programming. I incorporated this idea in the [High Performance
Computing](https://www.moreno.marzolla.name/teaching/hpc/) course that
I have been teaching over the last few years for the Computer science
and Engineering degree at the University of Bologna.

The course is an elective, undergraduate-level course in parallel
programming on shared-memory, distributed-memory and GPU
architectures. A considerable emphasis is put on practical aspects of
parallel programming using OpenMP, MPI and CUDA, for which suitable
exercises need to be developed.

This repository contains the source code of programming exercises that
are used in the lab sessions. The labs are organized as follows: each
exercise includes a detailed specification and a working serial
implementation. The goal is to parallelize the serial program using
one of the technologies that have been introduced in the previous
class. Solutions are made available at the end of each lab session.

Some exercises are quite simple, while others are more
complex. However, the level of difficulty is low since students are
expected to solve at least one exercise during each lab session.

Some notable points:

- These exercises require little or no knowledge outside parallel
  programming; in particular, they do not require advanced knowledge
  of physics, linear algebra or numerical analysis.

- Many exercises are designed to be interesting, and are taken from
  different domains such as 3D graphics, Cellular Automata,
  gravitational N-body solvers, cryptography and so on. Some programs
  produce images or movies as output, to make them more appealing.

- Some exercises can be parallelized using multiple programming
  paradigms.  This is quite useful to appreciate the strengths and
  weaknesses of each paradigm.

## Citation

The teaching methodology behind this collection of parallel
programming exercises has been described in the following paper:

> Moreno Marzolla, _Etudes for Parallel Programmers_, proc. 33rd
> Euromicro International Conference on Parallel, Distributed, and
> Network-Based Processing (PDP), march 12--14 2025, Turin, Italy,
> pp. 341--348, IEEE Computer Society Conference Publishing Services
> (CPS) ISBN: 979-8-3315-2493-7 ISSN: 2377-5750

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
  publisher = "IEEE CPS"
}
```

## List of exercises

Table 1 lists, for each exercise, which parallel versions are
available, and which parallel programming patterns are used to solve
it.

Table 1: List of exercises

| Kernel                      | OpenMP | MPI | CUDA | OpenCL | Pattern                                            |
|-----------------------------|--------|-----|------|--------|----------------------------------------------------|
| Password cracking           | X      | X   |      |        | Embarrassingly parallel                            |
| Dot product                 | X      | X   | X    | X      | Reduction, Scatter/Gather                          |
| Array sum                   |        | X   |      |        | Reduction, Scatter/Gather                          |
| Monte Carlo Pi              | X      | X   |      |        | Embarrassingly parallel, Reduction                 |
| Sieve of Eratosthenes       | X      |     | X    | X      | Embarrassingly parallel, Reduction                 |
| Character frequencies       | X      | X   | X    | X      | Embarrassingly parallel, Reduction                 |
| Inclusive scan              | X      | X   |      |        | Scan                                               |
| Dynamic task scheduling     | X      |     |      |        | Master-Worker                                      |
| MergeSort                   | X      |     |      |        | Task-level parallelism                             |
| Binary Tree traversal       | X      |     |      |        | Task-level parallelism                             |
| Ray tracing                 | X      | X   |      |        | Embarrassingly parallel, Scatter/Gather            |
| Levenstein's distance       | X      |     |      |        | 2D stencil, wavefront                              |
| Arnold's cat map            | X      |     | X    | X      | Embarrassingly parallel                            |
| Mandelbrot set              | X      | X   | X    | X      | Embarrassingly parallel, Load balancing            |
| Area of the Mandelbrot set  | X      | X   |      | X      | Embarrassingly parallel, Load balancing, Reduction |
| Image denoising             | X      |     | X    | X      | 2D Stencil                                         |
| List ranking                | X      |     |      |        | Pointer Jumping                                    |
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
| All-Pairs Shortest Paths    | X      |     |      |        | Embarrassingly parallel                            |

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
use the `SERIAL` preprocessor symbol: this symbol is defined when
compiling the skeleton, and is not defined when compiling the
solution.

```C
int foo(int x)
{
#ifdef SERIAL
   /* This block will be included in the serial skeleton provided
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
  plus any additional data file, are also copied there.

- The source code of the solution, placed into the `solutions/`
  subdirectory.

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

