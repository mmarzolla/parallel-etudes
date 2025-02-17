/******************************************************************************
 * mpi-c-ray - Ray tracing
 *
 * Copyright (C) 2006 John Tsiombikas <nuclear@siggraph.org>
 * Copyright (C) 2016, 2017, 2018, 2020, 2021, 2022, 2024 Moreno Marzolla
 *
 * You are free to use, modify and redistribute this program under the
 * terms of the GNU General Public License v2 or (at your option) later.
 * see "http://www.gnu.org/licenses/gpl.txt" for details.
 * ---------------------------------------------------------------------------
 * Usage:
 *   compile:  mpicc -std=c99 -Wall -Wpedantic -O2 mpi-c-ray.c -o mpi-c-ray -lm
 *   run:      mpirun -n 4 ./mpi-c-ray -s 1280x1024 -i sphfract.small.in -o sphfract.ppm
 *   convert:  convert sphfract.ppm sphfract.jpeg
 * ---------------------------------------------------------------------------
 * Scene file format:
 *   # sphere (many)
 *   s  x y z  rad   r g b   shininess   reflectivity
 *   # light (many)
 *   l  x y z
 *   # camera (one)
 *   c  x y z  fov_deg   targetx targety targetz
 ******************************************************************************/

/***
% HPC - Ray Tracing
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-10-05

The file [mpi-c-ray.c](mpi-c-ray.c) contains the implementation of a
[simple ray tracing program](https://github.com/jtsiomb/c-ray) written
by [John Tsiombikas](http://nuclear.mutantstargoat.com/) and released
under the GPLv2+ license. The instructions for compilation and use are
included in the comments. Some input files are provided, and should
produce the images shown in Figure 1.

![Figure 1: Some images produced by the program; the input files are,
from left to right: [sphfract.small.in](sphfract.small.in),
[spheres.in](spheres.in), [dna.in](dna.in)](omp-c-ray-images.png)

Table 1 shows the approximate time (in seconds) needed on my PC
(i7-4790 3.60GHz) to render each file using one core. The server is
slower because it has a lower clock frequency, but it has many cores
so the performance of the parallel version should be much better.

:Table 1: Render time using default parameters, single core Intel i7-4790 3.60GHz, gcc 7.5.0

File                                     Time (s)
---------------------------------------- ----------
[sphfract.big.in](sphfract.big.in)       478
[sphfract.small.in](sphfract.small.in)   19
[spheres.in](spheres.in)                 15
[dna.in](dna.in)                         9
---------------------------------------- ----------

The purpose of this exercise is to develop an MPI version of the
program; I suggest to proceed as follows:

- All MPI processes handle the command line as done in the provided
  serial program; in particular, you can assume that all MPI processes
  can read the input file. This assumption greatly simplifies the
  program since there is no need for the master to broadcast the input
  scene to all other processes.

- The output is written by process 0 only.

- Let $P$ be the number of MPI processes. Then, the image is
  partitioned in $P$ blocks. You can initially assume that the
  vertical size of the image is an integer multiple of $P$, and then
  relax this assumption.

- Each MPI process renders the assigned portion of the image. To do
  this, modify the function `render()` to receive two additional input
  parameters, `from` and` to`, with the meaning that the portion of
  image that must be rendered goes from line `from` (inclusive) to
  line `to` (excluded). To save memory, the `fb` parameter of function
  `render()` should point to a block of memory capable of holding
  `(to-from) * xsz` pixels.

- At the end, process 0 concatenates the blocks rendered by all
  processes (including itself). For this purpose, use `MPI_Gather()`
  or `MPI_Gatherv()`.

- Care should be put on memory management. All processes must have a
  local render buffer, whose size is calculate as above; the master
  has an additional buffer to hold the entire image.

Compile with:

        mpicc -std=c99 -Wall -Wpedantic mpi-c-ray.c -o mpi-c-ray -lm

To render the scene [sphfract.small.in](sphfract.small.in) you can
issue the command:

        mpirun -n 4 ./mpi-c-ray -s 800x600 < sphfract.small.in > img.ppm

This produces the image `img.ppm` of size $800 \times 600$. To display
the image under Windows it might be necessary to convert the format
to, e.g., JPEG with the command

        convert img.ppm img.jpeg

and then transfer `img.jpeg` on your Windows machine.

The program `mpi-c-ray` accepts some optional command-line parameters;
to get more information you can invoke the help:

        ./mpi-c-ray -h

It might be helpful to know the basics of [how a ray tracer
works](https://en.wikipedia.org/wiki/Ray_tracing_(graphics)) based on
[Whitted recursive
algorithm](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.156.1534)
(Figure 2).

![Figure 2: Operation diagram of a recursive ray tracer](omp-c-ray.png)

The scene is represented by a set of geometric primitives (spheres, in
our case). We generate a _primary ray_ (_V_) from the observer towards
each pixel. For each ray we determine the intersections (if any) with
the spheres in the scene. The point of intersection _p_ that is
closest to the observer is selected, and one or more _secondary rays_
are cast, depending on the material of the object _p_ belongs to:

- a _light ray_ (_L_) in the direction of each of the light sources;
  for each ray we compute intersections with the spheres to see
  whether _p_ is directly illuminated;

- if the surface of _p_ is reflective, we generate a _reflected ray_
  (_R_) and repeat recursively the procedure;

- if the surface is translucent, we generate a _transmitted ray_ (_T_)
  and repeat recursively the procedure (`omp-c-ray` does not support
  translucent objects, so this case never happens).

The time required to compute the color of a pixel depends, among other
things, on the number of spheres and lights, and on the material of
the spheres, and whether the primary ray intersects a sphere and
reflected rays are cast or not. This suggests that there could be a
high variability in the time required to compute the color of each
pixel, which leads to load imbalance that should be addressed in some
way.

## Files

- [mpi-c-ray.c](mpi-c-ray.c)
- [sphfract.small.in](sphfract.small.in) and [sphfract.big.in](sphfract.big.in) (generated by [genfract.c](genfract.c))
- [spheres.in](spheres.in) (generated by [genspheres.c](genspheres.c))
- [dna.in](dna.in) (generated by [gendna.c](gendna.c))

***/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <stdint.h> /* for uint8_t */
#include <assert.h>
#include <mpi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double x, y, z;
} vec3_t;

typedef struct {
    vec3_t orig, dir;
} ray_t;

typedef struct {
    vec3_t col;         /* color */
    double spow;	/* specular power */
    double refl;	/* reflection intensity */
} material_t;

typedef struct sphere {
    vec3_t pos;
    double rad;
    material_t mat;
    struct sphere *next;
} sphere_t;

typedef struct {
    vec3_t pos, normal, vref;	/* position, normal and view reflection */
    double dist;		/* parametric distance of intersection along the ray */
} spoint_t;

typedef struct {
    vec3_t pos, targ;
    double half_fov_rad;        /* half field of view in radiants */
} camera_t;

/* The __attribute__(( ... )) definition is gcc-specific, and tells
   the compiler that the fields of this structure should not be padded
   or aligned in any way. Since the structure only contains unsigned
   chars, it _might_ be unpadded by default; I am not sure,
   however. */
typedef struct __attribute__((__packed__)) {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* forward declarations */
vec3_t trace(ray_t ray, int depth);
vec3_t shade(sphere_t *obj, spoint_t *sp, int depth);

#define MAX_LIGHTS	16		/* maximum number of lights     */
const double RAY_MAG = 1000.0;		/* trace rays of this magnitude */
const int MAX_RAY_DEPTH	= 5;		/* raytrace recursion limit     */
const double ERR_MARGIN	= 1e-6;		/* an arbitrary error margin to avoid surface acne */
const double DEG_TO_RAD = M_PI / 180.0; /* convert degrees to radians   */

/* global state */
int xres = 800;
int yres = 600;
double aspect = 1.333333;
sphere_t *obj_list = NULL;
vec3_t lights[MAX_LIGHTS];
int lnum = 0; /* number of lights */
camera_t cam;

#define NRAN	1024
#define MASK	(NRAN - 1)
vec3_t urand[NRAN];
int irand[NRAN];

const char *usage = {
    "\n"
    "Usage: mpi-c-ray [options]\n\n"
    "  Reads a scene file from stdin, writes the image to stdout\n"
    "  and stats to stderr.\n\n"
    "Options:\n"
    "  -s WxH     width (W) and height (H) of the image (default 800x600)\n"
    "  -r <rays>  shoot <rays> rays per pixel (antialiasing, default 1)\n"
    "  -i <file>  read from <file> instead of stdin\n"
    "  -o <file>  write to <file> instead of stdout\n"
    "  -h         this help screen\n\n"
};

/* vector dot product */
double dot(vec3_t a, vec3_t b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


/* square of x */
double sq(double x)
{
    return x*x;
}


vec3_t normalize(vec3_t v)
{
    const double len = sqrt(dot(v, v));
    vec3_t result = v;
    result.x /= len;
    result.y /= len;
    result.z /= len;
    return result;
}


/* calculate reflection vector */
vec3_t reflect(vec3_t v, vec3_t n)
{
    vec3_t res;
    double d = dot(v, n);
    res.x = -(2.0 * d * n.x - v.x);
    res.y = -(2.0 * d * n.y - v.y);
    res.z = -(2.0 * d * n.z - v.z);
    return res;
}


vec3_t cross_product(vec3_t v1, vec3_t v2)
{
    vec3_t res;
    res.x = v1.y * v2.z - v1.z * v2.y;
    res.y = v1.z * v2.x - v1.x * v2.z;
    res.z = v1.x * v2.y - v1.y * v2.x;
    return res;
}


/* jitter function taken from Graphics Gems I. */
vec3_t jitter(int x, int y, int s)
{
    vec3_t pt;
    pt.x = urand[(x + (y << 2) + irand[(x + s) & MASK]) & MASK].x;
    pt.y = urand[(y + (x << 2) + irand[(y + s) & MASK]) & MASK].y;
    return pt;
}


/*
 * Compute ray-sphere intersection, and return {1, 0} meaning hit or
 * no hit.  Also the surface point parameters like position, normal,
 * etc are returned through the sp pointer if it is not NULL.
 */
int ray_sphere(const sphere_t *sph, ray_t ray, spoint_t *sp)
{
    double a, b, c, d, sqrt_d, t1, t2;

    a = sq(ray.dir.x) + sq(ray.dir.y) + sq(ray.dir.z);
    b = 2.0 * ray.dir.x * (ray.orig.x - sph->pos.x) +
        2.0 * ray.dir.y * (ray.orig.y - sph->pos.y) +
        2.0 * ray.dir.z * (ray.orig.z - sph->pos.z);
    c = sq(sph->pos.x) + sq(sph->pos.y) + sq(sph->pos.z) +
        sq(ray.orig.x) + sq(ray.orig.y) + sq(ray.orig.z) +
        2.0 * (-sph->pos.x * ray.orig.x - sph->pos.y * ray.orig.y - sph->pos.z * ray.orig.z) - sq(sph->rad);

    if ((d = sq(b) - 4.0 * a * c) < 0.0)
        return 0;

    sqrt_d = sqrt(d);
    t1 = (-b + sqrt_d) / (2.0 * a);
    t2 = (-b - sqrt_d) / (2.0 * a);

    if ((t1 < ERR_MARGIN && t2 < ERR_MARGIN) || (t1 > 1.0 && t2 > 1.0))
        return 0;

    if (sp) {
        if (t1 < ERR_MARGIN) t1 = t2;
        if (t2 < ERR_MARGIN) t2 = t1;
        sp->dist = t1 < t2 ? t1 : t2;

        sp->pos.x = ray.orig.x + ray.dir.x * sp->dist;
        sp->pos.y = ray.orig.y + ray.dir.y * sp->dist;
        sp->pos.z = ray.orig.z + ray.dir.z * sp->dist;

        sp->normal.x = (sp->pos.x - sph->pos.x) / sph->rad;
        sp->normal.y = (sp->pos.y - sph->pos.y) / sph->rad;
        sp->normal.z = (sp->pos.z - sph->pos.z) / sph->rad;

        sp->vref = reflect(ray.dir, sp->normal);
        sp->vref = normalize(sp->vref);
    }
    return 1;
}


vec3_t get_sample_pos(int x, int y, int sample)
{
    vec3_t pt;
    static double sf = 0.0;

    if (sf == 0.0) {
        sf = 2.0 / (double)xres;
    }

    pt.x = ((double)x / (double)xres) - 0.5;
    pt.y = -(((double)y / (double)yres) - 0.65) / aspect;

    if (sample) {
        vec3_t jt = jitter(x, y, sample);
        pt.x += jt.x * sf;
        pt.y += jt.y * sf / aspect;
    }
    return pt;
}


/* determine the primary ray corresponding to the specified pixel (x, y) */
ray_t get_primary_ray(int x, int y, int sample)
{
    ray_t ray;
    float m[3][3];
    vec3_t i, j = {0, 1, 0}, k, dir, orig, foo;

    k.x = cam.targ.x - cam.pos.x;
    k.y = cam.targ.y - cam.pos.y;
    k.z = cam.targ.z - cam.pos.z;
    k = normalize(k);

    i = cross_product(j, k);
    j = cross_product(k, i);
    m[0][0] = i.x; m[0][1] = j.x; m[0][2] = k.x;
    m[1][0] = i.y; m[1][1] = j.y; m[1][2] = k.y;
    m[2][0] = i.z; m[2][1] = j.z; m[2][2] = k.z;

    ray.orig.x = ray.orig.y = ray.orig.z = 0.0;
    ray.dir = get_sample_pos(x, y, sample);
    ray.dir.z = 1.0 / cam.half_fov_rad;
    ray.dir.x *= RAY_MAG;
    ray.dir.y *= RAY_MAG;
    ray.dir.z *= RAY_MAG;

    dir.x = ray.dir.x + ray.orig.x;
    dir.y = ray.dir.y + ray.orig.y;
    dir.z = ray.dir.z + ray.orig.z;
    foo.x = dir.x * m[0][0] + dir.y * m[0][1] + dir.z * m[0][2];
    foo.y = dir.x * m[1][0] + dir.y * m[1][1] + dir.z * m[1][2];
    foo.z = dir.x * m[2][0] + dir.y * m[2][1] + dir.z * m[2][2];

    orig.x = ray.orig.x * m[0][0] + ray.orig.y * m[0][1] + ray.orig.z * m[0][2] + cam.pos.x;
    orig.y = ray.orig.x * m[1][0] + ray.orig.y * m[1][1] + ray.orig.z * m[1][2] + cam.pos.y;
    orig.z = ray.orig.x * m[2][0] + ray.orig.y * m[2][1] + ray.orig.z * m[2][2] + cam.pos.z;

    ray.orig = orig;
    ray.dir.x = foo.x + orig.x;
    ray.dir.y = foo.y + orig.y;
    ray.dir.z = foo.z + orig.z;

    return ray;
}


/*
 * Compute direct illumination with the phong reflectance model.  Also
 * handles reflections by calling trace again, if necessary.
 */
vec3_t shade(sphere_t *obj, spoint_t *sp, int depth)
{
    vec3_t col = {0, 0, 0};

    /* for all lights ... */
    for (int i=0; i<lnum; i++) {
        double ispec, idiff;
        vec3_t ldir;
        ray_t shadow_ray;
        sphere_t *iter;
        int in_shadow = 0;

        ldir.x = lights[i].x - sp->pos.x;
        ldir.y = lights[i].y - sp->pos.y;
        ldir.z = lights[i].z - sp->pos.z;

        shadow_ray.orig = sp->pos;
        shadow_ray.dir = ldir;

        /* shoot shadow rays to determine if we have a line of sight
           with the light */
        for (iter = obj_list;
             (iter != NULL) && !ray_sphere(iter, shadow_ray, 0);
             iter = iter->next) {
            /* empty body */
        }
        in_shadow = (iter != NULL);
        /* and if we're not in shadow, calculate direct illumination
           with the phong model. */
        if (!in_shadow) {
            ldir = normalize(ldir);

            idiff = fmax(dot(sp->normal, ldir), 0.0);
            ispec = obj->mat.spow > 0.0 ? pow(fmax(dot(sp->vref, ldir), 0.0), obj->mat.spow) : 0.0;

            col.x += idiff * obj->mat.col.x + ispec;
            col.y += idiff * obj->mat.col.y + ispec;
            col.z += idiff * obj->mat.col.z + ispec;
        }
    }

    /* Also, if the object is reflective, spawn a reflection ray, and
       call trace() to calculate the light arriving from the mirror
       direction. */
    if (obj->mat.refl > 0.0) {
        ray_t ray;
        vec3_t rcol;

        ray.orig = sp->pos;
        ray.dir = sp->vref;
        ray.dir.x *= RAY_MAG;
        ray.dir.y *= RAY_MAG;
        ray.dir.z *= RAY_MAG;

        rcol = trace(ray, depth + 1);
        col.x += rcol.x * obj->mat.refl;
        col.y += rcol.y * obj->mat.refl;
        col.z += rcol.z * obj->mat.refl;
    }

    return col;
}


/*
 * trace a ray throught the scene recursively (the recursion happens
 * through shade() to calculate reflection rays if necessary).
 */
vec3_t trace(ray_t ray, int depth)
{
    vec3_t col;
    spoint_t sp, nearest_sp;
    sphere_t *nearest_obj = NULL;

    nearest_sp.dist = INFINITY;

    /* if we reached the recursion limit, bail out */
    if (depth >= MAX_RAY_DEPTH) {
        col.x = col.y = col.z = 0.0;
        return col;
    }

    /* find the nearest intersection ... */
    for (sphere_t *iter = obj_list; iter != NULL; iter = iter->next ) {
        if ( ray_sphere(iter, ray, &sp) &&
             (!nearest_obj || sp.dist < nearest_sp.dist) ) {
            nearest_obj = iter;
            nearest_sp = sp;
        }
    }

    /* and perform shading calculations as needed by calling shade() */
    if (nearest_obj != NULL) {
        col = shade(nearest_obj, &nearest_sp, depth);
    } else {
        col.x = col.y = col.z = 0.0;
    }

    return col;
}


/* render a frame of xsz/ysz dimensions into the provided framebuffer */
void render(int xsz, int ysz, int from, int to, pixel_t *fb, int samples)
{
    /*
     * for each subpixel, trace a ray through the scene, accumulate
     * the colors of the subpixels of each pixel, then put the colors
     * into the framebuffer.
     */
    for (int j=from, fb_row=0; j<to; j++, fb_row++) {
        for (int i=0; i<xsz; i++) {
            double r, g, b;
            r = g = b = 0.0;

            for (int s=0; s<samples; s++) {
                vec3_t col = trace(get_primary_ray(i, j, s), 0);
                r += col.x;
                g += col.y;
                b += col.z;
            }

            r /= samples;
            g /= samples;
            b /= samples;

            fb[fb_row*xsz+i].r = (uint8_t)(fmin(r, 1.0) * 255.0);
            fb[fb_row*xsz+i].g = (uint8_t)(fmin(g, 1.0) * 255.0);
            fb[fb_row*xsz+i].b = (uint8_t)(fmin(b, 1.0) * 255.0);
        }
    }
}


/* Load the scene from an extremely simple scene description file */
void load_scene(FILE *fp)
{
    char line[256], *ptr;

    obj_list = NULL;

    /* Default camera */
    cam.pos.x = cam.pos.y = cam.pos.z = 10.0;
    cam.half_fov_rad = 45 * DEG_TO_RAD * 0.5;
    cam.targ.x = cam.targ.y = cam.targ.z = 0.0;

    while ((ptr = fgets(line, sizeof(line), fp))) {
        int nread;
        sphere_t *sph;
        char type;
        double fov;

        while (*ptr == ' ' || *ptr == '\t') /* checking '\0' is implied */
            ptr++;
        if (*ptr == '#' || *ptr == '\n')
            continue;

        type = *ptr;
        ptr++;

        switch (type) {
        case 's': /* sphere */
            sph = malloc(sizeof *sph); assert(sph != NULL);
            sph->next = obj_list;
            obj_list = sph;

            nread = sscanf(ptr, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                           &(sph->pos.x), &(sph->pos.y), &(sph->pos.z),
                           &(sph->rad),
                           &(sph->mat.col.x), &(sph->mat.col.y), &(sph->mat.col.z),
                           &(sph->mat.spow), &(sph->mat.refl));
            assert(9 == nread);
            break;
        case 'l': /* light */
            if (lnum >= MAX_LIGHTS) {
                fprintf(stderr, "FATAL: too many lights\n");
                exit(-1);
            }
            nread = sscanf(ptr, "%lf %lf %lf",
                           &(lights[lnum].x),
                           &(lights[lnum].y),
                           &(lights[lnum].z));
            assert(3 == nread);
            lnum++;
            break;
        case 'c': /* camera */
            nread = sscanf(ptr, "%lf %lf %lf %lf %lf %lf %lf",
                           &cam.pos.x, &cam.pos.y, &cam.pos.z,
                           &fov,
                           &cam.targ.x, &cam.targ.y, &cam.targ.z);
            assert(7 == nread);
            cam.half_fov_rad = fov * DEG_TO_RAD * 0.5;
            break;
        default:
            fprintf(stderr, "unknown type: %c\n", type);
            abort();
        }
    }
}


/* Relinquish all memory used by the linked list of spheres */
void free_scene( void )
{
    while (obj_list != NULL) {
        sphere_t *next = obj_list->next;
        free(obj_list);
        obj_list = next;
    }
}


int main(int argc, char *argv[])
{
    pixel_t *pixels = NULL; /* where the global image is drawn (at rank 0) */
    pixel_t *local_pixels = NULL; /* where the local images are drawn */
    int rays_per_pixel = 1;
    FILE *infile = stdin, *outfile = stdout;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Every process parses the command line */
    for (int i=1; i<argc; i++) {
        if (argv[i][0] == '-' && argv[i][2] == 0) {
            char *sep;
            switch(argv[i][1]) {
            case 's':
                if (!isdigit(argv[++i][0]) || !(sep = strchr(argv[i], 'x')) || !isdigit(*(sep + 1))) {
                    fputs("-s must be followed by something like \"640x480\"\n", stderr);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                xres = atoi(argv[i]);
                yres = atoi(sep + 1);
                aspect = (double)xres / (double)yres;
                break;

            case 'i':
                if ((infile = fopen(argv[++i], "r")) == NULL) {
                    fprintf(stderr, "failed to open input file %s: %s\n", argv[i], strerror(errno));
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                break;

            case 'o':
                if ((outfile = fopen(argv[++i], "w")) == NULL) {
                    fprintf(stderr, "failed to open output file %s: %s\n", argv[i], strerror(errno));
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                break;

            case 'r':
                if (!isdigit(argv[++i][0])) {
                    fputs("-r must be followed by a number (rays per pixel)\n", stderr);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                rays_per_pixel = atoi(argv[i]);
                break;

            case 'h':
                fputs(usage, stdout);
                MPI_Finalize();
                return EXIT_SUCCESS;

            default:
                fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
                fputs(usage, stderr);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        } else {
            fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
            fputs(usage, stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    const int from = yres * my_rank / comm_sz;
    const int to = yres * (my_rank + 1) / comm_sz;

    if ((local_pixels = malloc(xres * (to - from) * sizeof(*pixels))) == NULL) {
        perror("local pixel buffer allocation failed");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    load_scene(infile);

    /* initialize the random number tables for the jitter */
    for (int i=0; i<NRAN; i++) urand[i].x = (double)rand() / RAND_MAX - 0.5;
    for (int i=0; i<NRAN; i++) urand[i].y = (double)rand() / RAND_MAX - 0.5;
    for (int i=0; i<NRAN; i++) irand[i] = (int)(NRAN * ((double)rand() / RAND_MAX));

    const double tstart = MPI_Wtime();
    render(xres, yres, from, to, local_pixels, rays_per_pixel);

    int counts[comm_sz], displs[comm_sz];
    for (int i=0; i<comm_sz; i++) {
        const int start = yres*i/comm_sz * xres * sizeof(pixel_t);
        const int end = yres*(i+1)/comm_sz * xres * sizeof(pixel_t);
        counts[i] = (end - start);
        displs[i] = start;
    }

    if ((my_rank == 0) &&
        (pixels = malloc(xres * yres * sizeof(*pixels))) == NULL) {
        perror("pixel buffer allocation failed");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Gather results from all nodes */
    MPI_Gatherv( local_pixels,  /* sendbuf */
                 counts[my_rank],/* sendcount */
                 MPI_BYTE,      /* sendtype */
                 pixels,        /* recvbuf */
                 counts,        /* receive counts */
                 displs,        /* displacements */
                 MPI_BYTE,      /* recvtype */
                 0,             /* root (where to send) */
                 MPI_COMM_WORLD /* communicator */
                 );

    const double elapsed = MPI_Wtime() - tstart;

    if (0 == my_rank) {
        /* output statistics to stderr */
        fprintf(stderr, "Rendering took %f seconds\n", elapsed);

        /* output the image */
        fprintf(outfile, "P6\n%d %d\n255\n", xres, yres);
        fwrite(pixels, sizeof(*pixels), xres*yres, outfile);
        fflush(outfile);
    }

    free(pixels);
    free(local_pixels);
    free_scene( );

    if (infile != stdin) fclose(infile);
    if (outfile != stdout) fclose(outfile);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
