/******************************************************************************
 * omp-c-ray - Ray tracing
 *
 * Copyright (C) 2006 John Tsiombikas <nuclear@siggraph.org>
 * Copyright (C) 2016, 2017, 2018, 2020, 2021, 2022 Moreno Marzolla <moreno.marzolla@unibo.it>
 *
 * You are free to use, modify and redistribute this program under the
 * terms of the GNU General Public License v2 or (at your option) later.
 * see "http://www.gnu.org/licenses/gpl.txt" for details.
 * ---------------------------------------------------------------------------
 * Usage:
 *   compile:  gcc -std=c99 -Wall -Wpedantic -fopenmp -O2 -o omp-c-ray omp-c-ray.c -lm
 *   run:      ./omp-c-ray -s 1280x1024 < sphfract.small.in > sphfract.ppm
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
% HPC - Ray tracing
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-10-19

The file [omp-c-ray.c](omp-c-ray.c) contains the implementation of a
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

To compile:

        gcc -std = c99 -Wall -Wpedantic -fopenmp omp-c-ray.c -o omp-c-ray -lm

To render the scene [sphfract.small.in](sphfract.small.in):

        ./omp-c-ray -s 800x600 <sphfract.small.in> img.ppm

The command above produces an image `img.ppm` with a resolution $800
\times 500$. To view the image on Windows it is useful to convert it
to JPEG format using the command:

        convert img.ppm img.jpeg

and then transferring `img.jpeg` to your PC for viewing.

The `omp-c-ray` program accepts a number of optional command-line
parameters; use

        ./omp-c-ray -h

to see the complete list.

Parallelize the `render()` function using appropriate OpenMP
directives. The serial program is well structured: in particular,
functions don't modify global variables, so there are not hidden
dependences. If you have time, measure the _speedup_ and the _strong
scaling efficienty_ of the parallel version.

It might be helpful to know the basics of [how a ray tracer
works](https://en.wikipedia.org/wiki/Ray_tracing_(graphics)) based on
the [Whitted recursive
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

- [omp-c-ray.c](omp-c-ray.c)
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
#include <omp.h>

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
    "Usage: omp-c-ray [options]\n\n"
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
    int i;
    vec3_t col = {0, 0, 0};

    /* for all lights ... */
    for (i=0; i<lnum; i++) {
        double ispec, idiff;
        vec3_t ldir;
        ray_t shadow_ray;
        sphere_t *iter = obj_list;
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
    sphere_t *iter;

    nearest_sp.dist = INFINITY;

    /* if we reached the recursion limit, bail out */
    if (depth >= MAX_RAY_DEPTH) {
        col.x = col.y = col.z = 0.0;
        return col;
    }

    /* find the nearest intersection ... */
    for (iter = obj_list; iter != NULL; iter = iter->next ) {
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
void render(int xsz, int ysz, pixel_t *fb, int samples)
{
    int i, j;

    /*
     * for each subpixel, trace a ray through the scene, accumulate
     * the colors of the subpixels of each pixel, then put the colors
     * into the framebuffer.
     */
#ifndef SERIAL
    /* You might want to play with the "schedule" clause */

#pragma omp parallel for default(none) collapse(2) shared(samples, fb, xsz, ysz) schedule(dynamic, 32)
#endif
    for (j=0; j<ysz; j++) {
        for (i=0; i<xsz; i++) {
            double r, g, b;
            int s;
            r = g = b = 0.0;

            for (s=0; s<samples; s++) {
                vec3_t col = trace(get_primary_ray(i, j, s), 0);
                r += col.x;
                g += col.y;
                b += col.z;
            }

            r /= samples;
            g /= samples;
            b /= samples;

            fb[j*xsz+i].r = (uint8_t)(fmin(r, 1.0) * 255.0);
            fb[j*xsz+i].g = (uint8_t)(fmin(g, 1.0) * 255.0);
            fb[j*xsz+i].b = (uint8_t)(fmin(b, 1.0) * 255.0);
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
    int i;
    double tstart, elapsed;
    pixel_t *pixels; /* framebuffer (where the image is drawn) */
    int rays_per_pixel = 1;
    FILE *infile = stdin, *outfile = stdout;

    for (i=1; i<argc; i++) {
        if (argv[i][0] == '-' && argv[i][2] == 0) {
            char *sep;
            switch(argv[i][1]) {
            case 's':
                if (!isdigit(argv[++i][0]) || !(sep = strchr(argv[i], 'x')) || !isdigit(*(sep + 1))) {
                    fputs("-s must be followed by something like \"640x480\"\n", stderr);
                    return EXIT_FAILURE;
                }
                xres = atoi(argv[i]);
                yres = atoi(sep + 1);
                aspect = (double)xres / (double)yres;
                break;

            case 'i':
                if ((infile = fopen(argv[++i], "r")) == NULL) {
                    fprintf(stderr, "failed to open input file %s: %s\n", argv[i], strerror(errno));
                    return EXIT_FAILURE;
                }
                break;

            case 'o':
                if ((outfile = fopen(argv[++i], "w")) == NULL) {
                    fprintf(stderr, "failed to open output file %s: %s\n", argv[i], strerror(errno));
                    return EXIT_FAILURE;
                }
                break;

            case 'r':
                if (!isdigit(argv[++i][0])) {
                    fputs("-r must be followed by a number (rays per pixel)\n", stderr);
                    return EXIT_FAILURE;
                }
                rays_per_pixel = atoi(argv[i]);
                break;

            case 'h':
                fputs(usage, stdout);
                return EXIT_SUCCESS;

            default:
                fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
                fputs(usage, stderr);
                return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
            fputs(usage, stderr);
            return EXIT_FAILURE;
        }
    }

    if ((pixels = malloc(xres * yres * sizeof(*pixels))) == NULL) {
        perror("pixel buffer allocation failed");
        return EXIT_FAILURE;
    }
    load_scene(infile);

    /* initialize the random number tables for the jitter */
    for (i=0; i<NRAN; i++) urand[i].x = (double)rand() / RAND_MAX - 0.5;
    for (i=0; i<NRAN; i++) urand[i].y = (double)rand() / RAND_MAX - 0.5;
    for (i=0; i<NRAN; i++) irand[i] = (int)(NRAN * ((double)rand() / RAND_MAX));

    tstart = omp_get_wtime();
    render(xres, yres, pixels, rays_per_pixel);
    elapsed = omp_get_wtime() - tstart;

    /* output statistics to stderr */
    fprintf(stderr, "Rendering took %f seconds\n", elapsed);

    /* output the image */
    fprintf(outfile, "P6\n%d %d\n255\n", xres, yres);
    fwrite(pixels, sizeof(*pixels), xres*yres, outfile);
    fflush(outfile);

    free(pixels);
    free_scene( );

    if (infile != stdin) fclose(infile);
    if (outfile != stdout) fclose(outfile);
    return EXIT_SUCCESS;
}
