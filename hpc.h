/****************************************************************************
 *
 * hpc.h - Miscellaneous utility functions for the HPC course
 *
 * Copyright (C) 2017 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 * Last modified on 2024-11-29 by Moreno Marzolla
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 *
 * This header file provides a function `double hpc_gettime()` that
 * returns the elapsed time (in seconds) since "the epoch". The
 * function uses the timing routing of the underlying parallel
 * framework (OpenMP or MPI), if enabled; otherwise, the default is to
 * use the `clock_gettime()` function.
 *
 * IMPORTANT NOTE: `clock_gettime()` is a POSIX extension; therefore,
 * when compiling with gcc, you MUST add
 *
 * #if _XOPEN_SOURCE < 600
 * #define _XOPEN_SOURCE 600
 * #endif
 *
 * at the beginning of your program, BEFORE any other include.
 *
 ****************************************************************************/

#ifndef HPC_H
#define HPC_H

#if defined(_OPENMP)
#include <omp.h>
/******************************************************************************
 * OpenMP timing routines
 ******************************************************************************/
double hpc_gettime( void )
{
    return omp_get_wtime();
}

#elif defined(MPI_Init)
/******************************************************************************
 * MPI timing routines
 ******************************************************************************/
double hpc_gettime( void )
{
    return MPI_Wtime();
}

#else
/******************************************************************************
 * POSIX-based timing routines
 ******************************************************************************/
#if _XOPEN_SOURCE < 600
#error You must add "#define _XOPEN_SOURCE 600" at the very beginning of your source program
#endif
#include <time.h>

double hpc_gettime( void )
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts );
    return ts.tv_sec + ((double)ts.tv_nsec) / 1.0e9;
}
#endif

#ifdef __CUDACC__

#include <stdio.h>
#include <stdlib.h>

/* from https://gist.github.com/ashwin/2652488 */

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifndef NO_CUDA_CHECK_ERROR
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        abort();
    }
#endif
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifndef NO_CUDA_CHECK_ERROR
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        abort();
    }

    /* More careful checking. However, this will affect performance.
       Comment away if needed. */
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        abort();
    }
#endif
}

#endif

#endif
