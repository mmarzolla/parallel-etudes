/****************************************************************************
 *
 * simpleCL.h
 *
 * Simplified C API for OpenCL programming
 *
 * Copyright 2011 Oscar Amoros Huguet, Cristian Garcia Marin
 * Copytight 2013 Camil Demetrescu
 * Copyright 2021, 2022 Moreno Marzolla
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
 ****************************************************************************/
#ifndef SIMPLECL_H
#define SIMPLECL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 220

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

typedef struct {
    cl_platform_id      platform_id;
    cl_context          context;
    cl_device_id        device_id;
    cl_command_queue    queue;
    cl_program          program;
    int                 devNum;
} sclDevice;

typedef struct {
    cl_kernel kernel;
    char kernel_name[98];
} sclKernel;

typedef struct {
    int ndims;
    size_t sizes[3];
} sclDim;

extern size_t SCL_DEFAULT_WG_SIZE;
extern size_t SCL_DEFAULT_WG_SIZE1D;
extern size_t SCL_DEFAULT_WG_SIZE2D;
extern size_t SCL_DEFAULT_WG_SIZE3D;

/******************************************************************************
 **
 ** INITIALIZATION
 **
 ******************************************************************************/

/**
 * Initialize the default OpenCL device. The default device is either
 * the first device of the first platform, or the device specified
 * with the SCL_DEFAULT_DEVICE environment variable.  The value is an
 * integer starting from 0; devices are enumerated as they are
 * encountered.
 *
 * This function must be called before any other call to other scl
 * functions.
 */
void sclInitFromString( const char *source );

/**
 * Compile the program whose source code is in the file `filename`
 */
void sclInitFromFile(const char *filename );

/**
 * Shuts down the OpenCL device and frees all resources acquired by
 * this program. This is normally the last function that is called
 * before the program terminates.
 */
void sclFinalize( void );

/******************************************************************************
 **
 ** MEMORY MANAGEMENT
 **
 ******************************************************************************/

/**
 * Allocates a block of `size` bytes on the device; the type of memory
 * is specified by `mode`. Although the `mode` parameter is passed to
 * `clCreateBuffer`, the only useful values for this simple interface
 * are:
 *
 * CL_MEM_READ_WRITE : the memory object will be read or written by a kernel;
 *
 * CL_MEM_WRITE_ONLY : the memory object will be written by a kernel
 *
 * CL_MEM_READ_ONLY : the memory object will be read by a kernel
 *
 */
cl_mem sclMalloc( size_t size, cl_int mode );

/**
 * Allocate a block of `size` bytes in device memory, and fill the
 * block with a copy of host memory from `hostPointer`. See
 * `sclMalloc()` for the meaning of `mode`.
 */
cl_mem sclMallocCopy( size_t size, void* hostPointer, cl_int mode );

/**
 * Copy `size` bytes from `src` on device to `dest` on host.  This is
 * a _blocking_ operation: when this function returns, the caller has
 * the guarantee that the data is in the `dest` buffer.
 */
void sclMemcpyDeviceToHost( void *dest, const cl_mem src, size_t size);

/**
 * Copy `size` bytes beginning from `offset` for device buffer `src`
 * to buffer `dest` on host.  This is a _blocking_ operation: when
 * this function returns, the caller has the guarantee that the data
 * is in the `dest` buffer.
 */
void sclMemcpyDeviceToHostOffset( void *dest, const cl_mem src, size_t offset, size_t size);

/**
 * Copy `size` bytes from `src` on host to `dest` on device.  When
 * this function returns, the caller can reuse the memory* pointed to
 * by `src`; however, there is no guarantee that the data has already
 * been moved to the device (it might have been buffered internally by
 * the OpenCL implementation).
 */
void sclMemcpyHostToDevice( cl_mem dest, const void *src, size_t size);

void sclMemset( cl_mem dest, int val, size_t size );

/**
 * Release the memory object `buff`
 */
void sclFree( cl_mem buff );

/******************************************************************************
 **
 ** KERNEL MANAGEMENT
 **
 ******************************************************************************/
sclDim DIM0(void);
sclDim DIM1(size_t x);
sclDim DIM2(size_t x, size_t y);
sclDim DIM3(size_t x, size_t y, size_t z);

/**
 * Creates a `sclKernel` object corresponding to the function whose
 * name is `name`; the function must be present in the file loaded at
 * initialization time (i.e., the file whose name has been passed as a
 * parameter to `sclInit()`), and must be defined as `__global` so
 * that it is callable from the host.
 */
sclKernel sclCreateKernel( const char* name );

/**
 * Releases a kernel object.
 */
void sclReleaseKernel( sclKernel soft );

/**
 * Launch a kernel. This function returns when execution completes
 * on the device.
 *
 * Parameters:
 *
 * `kernel` the kernel to launch; must have been created with `sclCreateKernel()`
 * `global_work_size`
 * `local_work_size`
 */
void sclLaunchKernel( sclKernel kernel, const sclDim global_work_size, const sclDim local_work_size );

void sclEnqueueKernel( sclKernel kernel, const sclDim global_work_size, const sclDim local_work_size );

void sclSetArgsLaunchKernel( sclKernel kernel, const sclDim global_work_size, const sclDim local_work_size, const char* fmt, ... );

void sclSetArgsEnqueueKernel( sclKernel kernel, const sclDim global_work_size, const sclDim local_work_size, const char* fmt, ... );

/******************************************************************************
 **
 ** DEVICE MANAGEMENT
 **
 ******************************************************************************/

/**
 * Wait for completion of all pending operations on the queue.
 */
cl_int sclDeviceSynchronize( void );

/**
 * Print hardware status
 */
void sclPrintHardwareStatus( void );

const char *sclGetErrorString( cl_int err );

size_t sclRoundUp(size_t s, size_t m);

void sclWGSetup1D(size_t xsize, sclDim *grid, sclDim *block);

void sclWGSetup2D(size_t xsize, size_t ysize, sclDim *grid, sclDim *block);

void sclWGSetup3D(size_t xsize, size_t ysize, size_t zsize, sclDim *grid, sclDim *block);

#endif
