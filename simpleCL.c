/****************************************************************************
 *
 * simpleCL.c
 *
 * Simplified C API for OpenCL programming
 *
 * Copyright 2011 Oscar Amoros Huguet, Cristian Garcia Marin
 * Copyright 2013 Camil Demetrescu
 * Copyright 2021, 2023 Moreno Marzolla
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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <unistd.h> /* for usleep */
#include "simpleCL.h"

/***
% `simpleCL`: A Simple C Wrapper for OpenCL
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-01-21

`simpleCL` is a C wrapper for OpenCL. Its goal is to make OpenCL
programming less tedious, by hiding most of the burden associated with
the myriad of low-level calls that are necessary to setup the device,
create the kernels and run them.

`simpleCL` is based on the [package with the same
name](https://github.com/morousg/simple-opencl) developed by Oscar
Amoros Huguet and Cristian Garcia Marin. However, this version is an
almost complete rewrite of the original code base; some features have
been dropped, others have been added, and some have been changed in a
non-compatible way. Therefore, **this program is not compatible with
the original simpleCL**.

`simpleCL` has been developed to support the [High Performance
Computing](https://www.moreno.marzolla.name/teaching/HPC) course
taught by [Moreno Marzolla](https://www.moreno.marzolla.name/) at the
University of Bologna. The course moved away from CUDA, so a
significant amount of CUDA programs had to be ported to
OpenCL. `simpleCL` is loosely inspired to the CUDA API in order to
simplify such conversion.

`simpleCL` has some drawbacks that make it unsuitable for
general-purpose OpenCL programming:

- It exposes a limited ad constrained subset of OpenCL capabilities,
  namely, those that are relevant for the HPC course mentioned above;

- `simpleCL` performs extensive error checking in order to facilitate
  debugging. No provision has been made to make these checks optional
  (e.g., with a comple-time or run-time flag), so applications relying
  on `simpleCL` should expect some level of performance degradation.

# General overview

The five basic steps for OpenCL programming with `simpleCL` are:

1. Initialize an OpenCL-capable device and load a program

2. Allocate memory on the device

3. Copy input from host to device, if necessary

4. Launch a kernel

5. Copy results from device to host

6. Free memory on the device

7. Terminate OpenCL

Steps 2--6 can be repeated as necessary.

 ***/

/******************************************************************************
 **
 ** Static variables and local constants
 **
 ******************************************************************************/
#define ENV_VAR_NAME "SCL_DEFAULT_DEVICE"
#define ENV_VAR_NDEBUG "SCL_NDEBUG"
#define MAX_PLATFORMS 8
#define MAX_DEVICES 16
#define MAX_STR_LEN 1024
#define MAX_BUILD_LOG_LEN 8192

static int nkernels = 0; /* number of kernels enqueued so far */
static sclDevice *scl_dev = NULL;
static int sclDebugEnabled = 1;
size_t SCL_DEFAULT_WG_SIZE = -1;
size_t SCL_DEFAULT_WG_SIZE1D = -1;
size_t SCL_DEFAULT_WG_SIZE2D = -1;
size_t SCL_DEFAULT_WG_SIZE3D = -1;

enum {
    SCL_SEPARATOR = ':',
    SCL_VALUE = 'v',
    SCL_BUFFER = 'b',
    SCL_LOCALMEM = 'L',
    SCL_INT = 'd',
    SCL_LONG = 'l',
    SCL_FLOAT = 'f'
};

/******************************************************************************
 **
 ** Internal functions (not visible by the user)
 **
 ******************************************************************************/
static void sclDebug(const char *fmt, ...)
{
    if (!sclDebugEnabled) return;

    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

static void sclvPanic(const char *fmt, va_list ap)
{
    fprintf(stderr, "PANIC: ");
    vfprintf(stderr, fmt, ap);
    abort();
}

static void sclPanic(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    sclvPanic(fmt, ap);
    va_end(ap); // never reached
}

static void sclCheckError(cl_int err, const char *fmt, ...)
{
    if (err != CL_SUCCESS) {
        va_list ap;
        va_start(ap, fmt);
        sclvPanic(fmt, ap);
        va_end(ap); // never reached
    }
}

static void sclCheckDeviceInitialized( void )
{
    if (scl_dev == NULL) {
        sclPanic("Hardware not initialized. You must call sclInit() at the beginning of your program\n");
    }
}

static void sclPrintDeviceInfo(sclDevice *dev)
{
    cl_int err;
    char opencl_version[MAX_STR_LEN]; // opencl version string
    char vendor[MAX_STR_LEN];         // vendor string
    char deviceName[MAX_STR_LEN];     // device name
    cl_uint numberOfCores;            // number of cores of on a device
    cl_long amountOfMemory;           // amount of memory on a device
    cl_uint clockFreq;                // clock frequency of a device
    cl_ulong maxAllocatableMem;       // maximum allocatable memory
    cl_ulong localMem;                // local memory for a device
    cl_bool available;                // tells if device is available
    size_t device_wg_size;            // max number of work items in a
                                      // work group
    cl_uint device_wi_dimensions;     // number of work item dimensions
    size_t device_wi_sizes[3];        // work item sizes

    const int f = 23;

    sclCheckDeviceInitialized();

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_NAME\n");

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_VENDOR\n");

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_OPENCL_C_VERSION, sizeof(opencl_version),
                          opencl_version, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_OPENCL_C_VERSION\n");

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numberOfCores),
                          &numberOfCores, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_MAX_COMPUTE_UNITS\n");

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(amountOfMemory),
                          &amountOfMemory, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_GLOBAL_MEM_SIZE\n");

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFreq),
                          &clockFreq, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_MAX_CLOCK_FREQUENCY\n");

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAllocatableMem),
                          &maxAllocatableMem, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_MAX_MEM_ALLOC_SIZE\n");

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem),
                          &localMem, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_LOCAL_MEM_SIZE\n");

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_AVAILABLE, sizeof(available),
                          &available, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_AVAILABLE\n");

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_wg_size),
                          &device_wg_size, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_MAX_WORK_GROUP_SIZE\n");

    err = clGetDeviceInfo(dev->device_id,
                          CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(device_wi_dimensions),
                          &device_wi_dimensions, NULL);
    sclCheckError(err, "failed to retrieve CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS\n");

    if (device_wi_dimensions == 3) {
        err = clGetDeviceInfo(dev->device_id,
                              CL_DEVICE_MAX_WORK_ITEM_SIZES,
                              sizeof(device_wi_sizes),
                              device_wi_sizes, NULL);
        sclCheckError(err, "failed to retrieve CL_DEVICE_MAX_WORK_ITEM_SIZES\n");
    }

    // print out device info
    sclDebug( "\n\nInfo for device %d:\n\n", dev->devNum);
    sclDebug( "%-*s %s\n", f, "Name:", deviceName);
    sclDebug( "%-*s %s\n", f, "Vendor:", vendor);
    sclDebug( "%-*s %s\n", f, "OpenCL version:", opencl_version);
    sclDebug( "%-*s %s\n", f, "Available:", available ? "Yes" : "No");
    sclDebug( "%-*s %u\n", f, "Compute units:", numberOfCores);
    sclDebug( "%-*s %u MHz\n", f, "Clock frequency:", clockFreq);
    sclDebug( "%-*s %0.00f MB\n", f, "Global memory:", (double)amountOfMemory/1048576);
    sclDebug( "%-*s %0.00f MB\n", f, "Max allocatable memory:", (double)maxAllocatableMem/1048576);
    sclDebug( "%-*s %u KB\n", f, "Local memory:", (unsigned int)localMem);
    sclDebug( "%-*s %lu\n", f, "Max work group size:", device_wg_size);
    if (device_wi_dimensions == 3) {
        sclDebug( "%-*s %d,%d,%d\n", f, "Max work iten sizes:", (int)device_wi_sizes[0], (int)device_wi_sizes[1], (int)device_wi_sizes[2]);
    }
    sclDebug( "\n\n" );
}

static int sclGetDefaultDeviceNo( int ndevices )
{
    const char* env = getenv(ENV_VAR_NAME);
    int device = 0;
    if (env != NULL) {
        device = atoi(env);
        if (device < 0 || device >= ndevices) {
            fprintf(stderr, "\n%s environment variable set to invalid value %d; defaulting to 0\n", ENV_VAR_NAME, device);
            device = 0;
        }
    }
    return device;
}

static void sclPrintDeviceNamePlatforms( sclDevice* devices, int n, int default_dev_no )
{
    cl_char deviceName[MAX_STR_LEN];
    cl_char platformVendor[MAX_STR_LEN];
    cl_char platformName[MAX_STR_LEN];

    for ( int i = 0; i < n; ++i ) {
        clGetPlatformInfo( devices[i].platform_id, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL );
        clGetPlatformInfo( devices[i].platform_id, CL_PLATFORM_VENDOR, sizeof(platformVendor), platformVendor, NULL );
        clGetDeviceInfo( devices[i].device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL );
        sclDebug("Device %d%s\n", devices[i].devNum,
                 (i == default_dev_no ? " USED" : ""));
        sclDebug("\tPlatform.. %s\n", platformName);
        sclDebug("\tVendor.... %s\n", platformVendor);
        sclDebug("\tDevice.... %s\n", deviceName );
    }
}



/******************************************************************************
 **
 ** Public functions
 **
 ******************************************************************************/
const char *sclGetErrorString( cl_int err )
{
    static struct {
        cl_int errcode;
        const char *msg;
    } errmessages[] = {
        {CL_DEVICE_NOT_FOUND, "CL_DEVICE_NOT_FOUND"},
        {CL_DEVICE_NOT_AVAILABLE, "CL_DEVICE_NOT_AVAILABLE"},
        {CL_COMPILER_NOT_AVAILABLE, "CL_COMPILER_NOT_AVAILABLE"},
        {CL_PROFILING_INFO_NOT_AVAILABLE, "CL_PROFILING_INFO_NOT_AVAILABLE"},
        {CL_MEM_COPY_OVERLAP, "CL_MEM_COPY_OVERLAP"},
        {CL_IMAGE_FORMAT_MISMATCH, "CL_IMAGE_FORMAT_MISMATCH"},
        {CL_IMAGE_FORMAT_NOT_SUPPORTED, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
        {CL_INVALID_COMMAND_QUEUE, "CL_INVALID_COMMAND_QUEUE"},
        {CL_INVALID_CONTEXT, "CL_INVALID_CONTEXT"},
        {CL_INVALID_MEM_OBJECT, "CL_INVALID_MEM_OBJECT"},
        {CL_INVALID_VALUE, "CL_INVALID_VALUE"},
        {CL_INVALID_EVENT_WAIT_LIST, "CL_INVALID_EVENT_WAIT_LIST"},
        {CL_MEM_OBJECT_ALLOCATION_FAILURE, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
        {CL_OUT_OF_HOST_MEMORY, "CL_OUT_OF_HOST_MEMORY"},
        {CL_INVALID_PROGRAM_EXECUTABLE, "CL_INVALID_PROGRAM_EXECUTABLE"},
        {CL_INVALID_KERNEL, "CL_INVALID_KERNEL"},
        {CL_INVALID_KERNEL_ARGS, "CL_INVALID_KERNEL_ARGS"},
        {CL_INVALID_WORK_DIMENSION, "CL_INVALID_WORK_DIMENSION"},
#ifndef __APPLE__
        {CL_INVALID_GLOBAL_WORK_SIZE, "CL_INVALID_GLOBAL_WORK_SIZE"},
#endif
        {CL_INVALID_WORK_GROUP_SIZE, "CL_INVALID_WORK_GROUP_SIZE"},
        {CL_INVALID_WORK_ITEM_SIZE, "CL_INVALID_WORK_ITEM_SIZE"},
        {CL_INVALID_GLOBAL_OFFSET, "CL_INVALID_GLOBAL_OFFSET"},
        {CL_OUT_OF_RESOURCES, "CL_OUT_OF_RESOURCES"},
        {CL_INVALID_PROGRAM, "CL_INVALID_PROGRAM"},
        {CL_INVALID_KERNEL_NAME, "CL_INVALID_KERNEL_NAME"},
        {CL_INVALID_KERNEL_DEFINITION, "CL_INVALID_KERNEL_DEFINITION"},
        {CL_INVALID_BUFFER_SIZE, "CL_INVALID_BUFFER_SIZE"},
        {CL_BUILD_PROGRAM_FAILURE, "CL_BUILD_PROGRAM_FAILURE"},
        {CL_INVALID_ARG_INDEX, "CL_INVALID_ARG_INDEX"},
        {CL_INVALID_ARG_VALUE, "CL_INVALID_ARG_VALUE"},
        {CL_MAP_FAILURE, "CL_MAP_FAILURE"},
        {CL_MISALIGNED_SUB_BUFFER_OFFSET, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
        {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
        {CL_INVALID_DEVICE_TYPE, "CL_INVALID_DEVICE_TYPE"},
        {CL_INVALID_PLATFORM, "CL_INVALID_PLATFORM"},
        {CL_INVALID_DEVICE, "CL_INVALID_DEVICE"},
        {CL_INVALID_QUEUE_PROPERTIES, "CL_INVALID_QUEUE_PROPERTIES"},
        {CL_INVALID_HOST_PTR, "CL_INVALID_HOST_PTR"},
        {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
        {CL_INVALID_IMAGE_SIZE, "CL_INVALID_IMAGE_SIZE"},
        {CL_INVALID_SAMPLER, "CL_INVALID_SAMPLER"},
        {CL_INVALID_BINARY, "CL_INVALID_BINARY"},
        {CL_INVALID_BUILD_OPTIONS, "CL_INVALID_BUILD_OPTIONS"},
        {CL_INVALID_ARG_SIZE, "CL_INVALID_ARG_SIZE"},
        {CL_INVALID_EVENT, "CL_INVALID_EVENT"},
        {CL_INVALID_OPERATION, "CL_INVALID_OPERATION"},
        {CL_INVALID_GL_OBJECT, "CL_INVALID_GL_OBJECT"},
        {CL_INVALID_MIP_LEVEL, "CL_INVALID_MIP_LEVEL"},
        {CL_INVALID_PROPERTY, "CL_INVALID_PROPERTY"},
        {0, NULL}
    };
    static char errMessageUnknown[MAX_STR_LEN];
    int i;

    for (i=0; (errmessages[i].errcode != err) &&
             (errmessages[i].msg != NULL); i++)
        { /* empty body */ }
    if (errmessages[i].msg != NULL) {
        return errmessages[i].msg;
    } else {
        snprintf(errMessageUnknown, sizeof(errMessageUnknown),
                 "Unknown error code %d", err);
        return errMessageUnknown;
    }
}

/******************************************************************************
 **
 ** INITIALIZATION
 **
 ******************************************************************************/
/***

# Initialization

Before using any of the functions provided by the `simpleCL`
interface, it is necessary to initialize an OpenCL capable device and
loading a program on the device. This can be done using either `void
sclInitFromString(const char *source)` or `void sclInitFromFile(const
char *filename)`.

Device selection works as follows:

- If the environment variable `SCL_DEFAULT_DEVICE` is set, its value
  is used as the index of the default device. OpenCL devices are
  enumerated in the order they are reported by the lower-level OpenCL
  calls, starting from device 0.

- Otherwise, if the environment variable `SCL_DEFAULT_DEVICE` is not
  set or is set to an invalid value, the first OpenCL device that is
  enumerated by the low-level OpenCL calls is the default device.

For debugging purposes, the list of devices found and some additional
information is printed to stderr.  These messages can be suppressed by
setting the environment variable `SCL_NDEBUG` to any value.

After initialization, the following symbols are defined for both host
and device code:

- `SCL_DEFAULT_WG_SIZE`
- `SCL_DEFAULT_WG_SIZE1D`
- `SCL_DEFAULT_WG_SIZE2D`
- `SCL_DEFAULT_WG_SIZE3D`

These have numeric (integer) values that represent the default maximum
size of a workgropup (`SCL_DEFAULT_WG_SIZE`), as returned by the
hardware. The other values are the size of 1D, 2D and 3D workgroups.

## `void sclInitFromString(const char *source)`

Initialize the default OpenCL device and loads the program whose
source code is in the string `source`.

This function may be called at most once.

 ***/
void sclInitFromString( const char *source )
{
    cl_uint nPlatforms = 0;
    cl_platform_id platforms[MAX_PLATFORMS];
    cl_device_id device_ids[MAX_DEVICES];
    static sclDevice devices[MAX_DEVICES];
    int dev_count = 0;

    if (scl_dev != NULL) {
        sclPanic( "Device already initialized; you can not call sclInitFromString()/sclInitFromFile() more than once\n" );
    }

    /* check if debug must be disabled */
    if (getenv(ENV_VAR_NDEBUG) != NULL) {
        sclDebugEnabled = 0;
    }

    cl_int err = clGetPlatformIDs( MAX_PLATFORMS, platforms, &nPlatforms );
    if (err != CL_SUCCESS || nPlatforms == 0) {
        sclPanic( "No OpenCL platform found\n");
    }

    /* Enumerate platforms and devices */
    for ( int p = 0; p < (int)nPlatforms; p++ ) {
        cl_uint nDevices = 0;
        err = clGetDeviceIDs( platforms[p], CL_DEVICE_TYPE_ALL, MAX_DEVICES, device_ids, &nDevices );
        if ( nDevices == 0 ) {
            fprintf(stderr, "No OpenCL enabled device found for platform %d\n", p);
        } else {
            for ( int d = 0; d < (int)nDevices; d++ ) {
                devices[ dev_count ].platform_id    = platforms[ p ];
                devices[ dev_count ].device_id      = device_ids[ d ];
                devices[ dev_count ].devNum         = dev_count;
                dev_count++;
            }
        }
    }
    const int default_dev_no = sclGetDefaultDeviceNo(dev_count);

    scl_dev = &devices[default_dev_no];
    sclPrintDeviceNamePlatforms(devices, dev_count, default_dev_no);
    sclPrintDeviceInfo(scl_dev);

    scl_dev->context = clCreateContext(0, 1, &scl_dev->device_id, NULL, NULL, &err);
    sclCheckError(err, "clCreateContext error in sclInitFromString: %s\n", sclGetErrorString(err));

    scl_dev->queue = clCreateCommandQueue(scl_dev->context, scl_dev->device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    sclCheckError(err, "clCreateCommandQueue error in sclInitFromStringt: %s\n", sclGetErrorString(err));

    scl_dev->program = clCreateProgramWithSource( scl_dev->context, 1, (const char**)&source, NULL, &err );
    sclCheckError(err, "clCreateProgramWithSource error in sclInitFromString: %s\n", sclGetErrorString(err));

    /* retrieve max workgroup size */
    err = clGetDeviceInfo(scl_dev->device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(SCL_DEFAULT_WG_SIZE), &SCL_DEFAULT_WG_SIZE, NULL);
    sclCheckError(err, "clGetDeviceInfo error while retrieving CL_DEVICE_MAX_WORK_GROUP_SIZE in sclInitFromSrring: %s\n", sclGetErrorString(err));

    /* compute sizes for 1D, 2D, 3D workgroups */
    SCL_DEFAULT_WG_SIZE1D = SCL_DEFAULT_WG_SIZE;

    for (SCL_DEFAULT_WG_SIZE2D = 1; SCL_DEFAULT_WG_SIZE2D*SCL_DEFAULT_WG_SIZE2D <= SCL_DEFAULT_WG_SIZE; SCL_DEFAULT_WG_SIZE2D *= 2) ;
    SCL_DEFAULT_WG_SIZE2D /= 2;

    for (SCL_DEFAULT_WG_SIZE3D = 1; SCL_DEFAULT_WG_SIZE3D*SCL_DEFAULT_WG_SIZE3D*SCL_DEFAULT_WG_SIZE3D <= SCL_DEFAULT_WG_SIZE; SCL_DEFAULT_WG_SIZE3D *= 2) ;
    SCL_DEFAULT_WG_SIZE3D /= 2;

    char build_options[MAX_STR_LEN];
    char build_log[MAX_BUILD_LOG_LEN];
    /*
      Pass the values of SCL_DEFAULT_WG_SIZExx as #define's
     */
    snprintf(build_options, sizeof(build_options),
             "-D SCL_DEFAULT_WG_SIZE=%d "
             "-D SCL_DEFAULT_WG_SIZE1D=%d "
             "-D SCL_DEFAULT_WG_SIZE2D=%d "
             "-D SCL_DEFAULT_WG_SIZE3D=%d",
             (int)SCL_DEFAULT_WG_SIZE,
             (int)SCL_DEFAULT_WG_SIZE1D,
             (int)SCL_DEFAULT_WG_SIZE2D,
             (int)SCL_DEFAULT_WG_SIZE3D);
    err = clBuildProgram( scl_dev->program, 0, NULL, build_options, NULL, NULL );
    if ( err != CL_SUCCESS ) {
        clGetProgramBuildInfo( scl_dev->program, scl_dev->device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL );
        fprintf(stderr, "\n\n----- COMPILATION ERROR for OpenCL program\n");
        fputs(build_log, stderr);
        fprintf(stderr, "----- END COMPILATION ERROR\n\n");
        sclPanic("clBuildProgram error in sclInitFromString: %s\n", sclGetErrorString(err));
    }
}

/***

## `void sclInitFromFile( const char *filename )`

Initialize the default OpenCL device and loads the program whose
source code is in file `filename`.

This function may be called at most once.

***/
void sclInitFromFile( const char *filename )
{
    FILE *f = fopen( filename, "r" );
    if ( f == NULL ) {
        sclPanic("Can not open program file \"%s\"\n", filename);
    }
    fseek(f, 0L, SEEK_END);
    const size_t size = ftell(f);
    rewind(f);
    char *source = (char *)malloc( size + 1 );
    assert(source != NULL);
    if( fread( source, 1, size, f ) != size ) {
        sclPanic("Error loading program file \"%s\"\n", filename);
    }
    source[ size ] = '\0';
    fclose( f );
    sclInitFromString(source);
    free(source);
}

/***

## `void sclFinalize(void)`

Waits for all pending operations on the device to complete, and clean
up the environment. After this function is called, no more OpenCL
commands can be issued.

 ***/
void sclFinalize( void )
{
    sclCheckDeviceInitialized();

    cl_int err;

    sclDeviceSynchronize();
    err = clReleaseProgram( scl_dev->program );
    sclCheckError(err, "clReleaseProgram in sclFinalize: %s\n", sclGetErrorString(err));

    err = clReleaseCommandQueue( scl_dev->queue);
    sclCheckError(err, "clReleaseCommandQueue in sclFinalize: %s\n", sclGetErrorString(err));

    err = clReleaseContext( scl_dev->context );
    sclCheckError(err, "clReleaseContext in sclFinalize: %s\n", sclGetErrorString(err));

    scl_dev = NULL;
}



/******************************************************************************
 **
 ** MEMORY MANAGEMENT
 **
 ******************************************************************************/

/***

# Memory management

## `cl_mem sclMalloc(size_t size, cl_int mode)

The function 'sclMalloc()` allocates `size` bytes on the current
device and returns the associated memory object. `mode` is a bitmask
that is used to specify the type of memory that must be allocated;
The supported values are:

- `CL_MEM_READ_WRITE`: the memory block will be read and written by
  kernels; this is probably the case in most situations.

- `CL_MEM_READ_ONLY`: the memory block will be read but not written by
  kernels (it can be written by the host, anyway).

- `CL_MEM_WRITE_ONLY`: the memory block will be written but not read
  by kernels (read attempts from kernel code will produce undefined
  behavior).

 ***/
cl_mem sclMalloc( size_t size, cl_int mode )
{
    sclCheckDeviceInitialized();

    cl_int err;
    cl_mem buffer = clCreateBuffer( scl_dev->context, mode, size, NULL, &err );
    sclCheckError(err, "clCreateBuffer error in sclMalloc: %s\n", sclGetErrorString(err));
    if ( buffer == NULL ) {
        sclPanic("clCreateBuffer error in sclMalloc: NULL returned");
    }
    return buffer;
}

/***

## `cl_mem sclMallocCopy(size_t size, void *hostPointer, cl_int mode)`

The function `sclMallocCopy() allocates a block of `size` bytes on the
device, and initializes it with the content of host memory starting
from the address `hostPointer`. See `sclMalloc()` for the meaning of
`mode`.

This function is locally equivalent to `sclMalloc()` followed by
`sclMemcpyHostToDevice()`; however, it might be more efficient to use
`sclMallocCopy()`, depending on the device.

***/
cl_mem sclMallocCopy( size_t size, void* hostPointer, cl_int mode )
{
    sclCheckDeviceInitialized();

    cl_int err;
    cl_mem buffer = clCreateBuffer( scl_dev->context, mode | CL_MEM_COPY_HOST_PTR, size, hostPointer, &err );
    sclCheckError(err, "clCreateBuffer error in sclMallocCopy: %s\n", sclGetErrorString(err));
    if ( buffer == NULL ) {
        sclPanic("clCreateBuffer error in sclMallocCopy: NULL returned");
    }
    return buffer;
}

/***

## `cl_mem sclCreateSubBuffer(cl_mem buffer, size_t origin, size_t size)`

Crate a sub-buffer of `buffer` that spans `size` bytes starting from
offset `origin`. Access modes of the sub-buffer (`CL_MEM_READ_ONLY`,
`CL_MEM_WRITE_ONLY` or `CL_MEM_READ_WRITE`) are inherited from those
of `buffer`.

Sub-buffers can be useful in cases when only a portion of a memory
region has to be transferred from host to device (or the other way
around).

 ***/
cl_mem sclCreateSubBuffer(cl_mem buffer, size_t origin, size_t size)
{
    struct _cl_buffer_region {
        size_t origin;
        size_t size;
    } region = {origin, size};
    cl_int err;

    cl_mem result = clCreateSubBuffer( buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    sclCheckError(err, "clCreateSubBuffer error in sclCreateSubBuffer: %s\n", sclGetErrorString(err));
    return result;
}

/***

## `void sclFree(cl_mem buf)`

Frees the memory buffer `buf` that was previously allocated with
`sclMalloc()` or `sclMallocCopy()`.

 ***/
void sclFree( cl_mem buf )
{
    sclCheckDeviceInitialized();

    const cl_int err = clReleaseMemObject( buf );
    sclCheckError(err, "clReleaseMemObject in sclFree: %s\n", sclGetErrorString(err));
}

/***

## `void sclMemcpyHostToDevice(cl_mem dest, const void *src, size_t size)`

Copies `size` bytes from host memory starting at address `src` to
device memory referred to by `dest`. The copy operation is
asynchronous: this function terminates as soon as the data transfer
command has been queued for execution, but the actual copy might take
place later on. However, it is safe to modify the content of the host
memory block `src` as soon as this function returns.

 ***/
void sclMemcpyHostToDevice( cl_mem dest, const void *src, size_t size )
{
    sclCheckDeviceInitialized();

    const cl_int err = clEnqueueWriteBuffer( scl_dev->queue, dest, CL_TRUE, 0, size, src, 0, NULL, NULL );
    sclCheckError(err, "clEnqueueWriteBuffer error in sclMemcpyHostToDevice: %s\n", sclGetErrorString(err));
}

/***

## `void sclMemcpyDeviceToHost(void *dest, cl_mem src, size_t size)`

Copies `size` bytes from the device memory buffer `src` to host memory
starting at address `dest`. Upon return, the destination buffer is
guaranteed to contain the data.

 ***/
void sclMemcpyDeviceToHost( void *dest, const cl_mem src, size_t size )
{
    sclCheckDeviceInitialized();

    const cl_int err = clEnqueueReadBuffer( scl_dev->queue, src, CL_TRUE, 0, size, dest, 0, NULL, NULL );
    sclCheckError(err, "clEnqueueReadBuffer error in sclMemcpyDeviceToHost: %s\n", sclGetErrorString(err));
}

/***

## `void sclMemcpyDeviceToHostOffset(void *dest, cl_mem src, size_t offset, size_t size)`

Copies `size` bytes from `offset` bytes from the beginning of device
memory buffer `src` to host memory starting at address `dest`. Upon
return, the destination buffer is guaranteed to contain the data.

 ***/
void sclMemcpyDeviceToHostOffset( void *dest, const cl_mem src, size_t offset, size_t size )
{
    sclCheckDeviceInitialized();

    const cl_int err = clEnqueueReadBuffer( scl_dev->queue, src, CL_TRUE, offset, size, dest, 0, NULL, NULL );
    sclCheckError(err, "clEnqueueReadBuffer error in sclMemcpyDeviceToHostOffset: %s\n", sclGetErrorString(err));
}

/***

## `void sclMemset(cl_mem dest, int val, size_t size)`

Fills the device memory buffer `dest` of size `size` with bytes
containing the value `val`. `val` is converted to `unsigned char`
before being written, as the standard C function `memset()` does.

***/
void sclMemset( cl_mem dest, int val, size_t size )
{
    sclCheckDeviceInitialized();
    /* OpenCL 1.2 provides a function clEuqueueFillBuffer for this
       purpose.  Unfortunately, old devices do not support it;
       therefore, to be on the safe side we employ a slower, less
       efficient general-purpose version that copies an initialized
       buffer from host to device memory.

       The proper way would be to query the device for OpenCL 1.2
       support, and use the most appropriate method accordingly. */
#if 0
    const unsigned char pattern = (unsigned char)val;
    const cl_int err = clEnqueueFillBuffer(scl_dev->queue,
                                           dest,
                                           &pattern,
                                           sizeof(pattern),
                                           0,
                                           size,
                                           0,
                                           NULL,
                                           NULL);
    sclCheckError(err, "clEnqueueFillBuffer error in sclMemset: %s\n", sclGetErrorString(err));
#else
    char *buf = (char*)malloc(size); assert(buf != NULL);
    for (int i=0; i<size; i++)
        buf[i] = val;
    sclMemcpyHostToDevice(dest, buf, size);
    free(buf);
#endif
}



/******************************************************************************
 **
 ** KERNEL MANAGEMENT
 **
 ******************************************************************************/

/***

# Kernel management



 ***/
sclDim DIM0(void)
{
    sclDim result;
    result.ndims = 0; /* means undefined */
    return result;
}

/***

## `sclDim DIM1(size_t x)`

Returns a new `sclDim` object representing a 1D block of size `x`.

 ***/
sclDim DIM1(size_t x)
{
    sclDim result;
    result.ndims = 1;
    result.sizes[0] = x;
    return result;
}

/***

## `sclDim DIM2(size_t x, size_t y)`

Returns a new `sclDim` object representing a 2D block of size `x`
by `y`.

 ***/
sclDim DIM2(size_t x, size_t y)
{
    sclDim result;
    result.ndims = 2;
    result.sizes[0] = x;
    result.sizes[1] = y;
    return result;
}

/***

## `sclDim DIM3(size_t x, size_t y, size_t z)`

Returns a new `sclDim` object representing a 3D block of size `x`
by `y` by `z`.

 ***/
sclDim DIM3(size_t x, size_t y, size_t z)
{
    sclDim result;
    result.ndims = 3;
    result.sizes[0] = x;
    result.sizes[1] = y;
    result.sizes[2] = z;
    return result;
}

/***

## `sclKernel sclCreateKernel(const char *name)`

Return a `sclKernel` object corresponding to function `name`, that
must be present in the device program loded with `sclInitFromString()`
or `sclInitFromFile()`; function `name` must be declared as a kernel
(using the `__kernel` or `kernel` keywords).

 ***/
sclKernel sclCreateKernel( const char* name )
{
    cl_int err;
    sclKernel kern;

    strncpy(kern.kernel_name, name, sizeof(kern.kernel_name));
    kern.kernel = clCreateKernel( scl_dev->program, kern.kernel_name, &err );
    sclCheckError(err, "clCrateKernel error in sclCreateKernel for kernel %s: %s\n", name, sclGetErrorString(err));
    return kern;
}

void sclReleaseKernel( sclKernel kern )
{
    sclCheckDeviceInitialized();

    const cl_int err = clReleaseKernel( kern.kernel );
    sclCheckError(err, "Error in clReleaseKernel called by sclReleaseKernel for kernel %s: %s\n", kern.kernel_name, sclGetErrorString(err));
}

void sclLaunchKernel( sclKernel kernel,
                      const sclDim global_work_size,
                      const sclDim local_work_size )
{
    sclCheckDeviceInitialized();
    assert( ((local_work_size.ndims == 0) && (global_work_size.ndims > 0)) ||
            (global_work_size.ndims == local_work_size.ndims) );
    const cl_int err =
            clEnqueueNDRangeKernel( scl_dev->queue,
                                    kernel.kernel,
                                    global_work_size.ndims,
                                    NULL, /* no offset */
                                    global_work_size.sizes,
                                    (local_work_size.ndims < 1 ? NULL : local_work_size.sizes),
                                    0, /* no events in wait list */
                                    NULL, /* empty wait list */
                                    NULL );
    sclCheckError(err, "clEnqueueNDRangeKernel error in sclLaunchKernel for kernel %s: %s\n", kernel.kernel_name, sclGetErrorString(err));
    sclDeviceSynchronize( );
    nkernels = 0;
}

void sclEnqueueKernel( sclKernel kernel,
                       const sclDim global_work_size,
                       const sclDim local_work_size )
{
    static const int SYNC_EVERY = 100; /* after how many operations to force a flush of the command queue */
    sclCheckDeviceInitialized();

    assert( ((local_work_size.ndims == 0) && (global_work_size.ndims > 0)) ||
            (global_work_size.ndims == local_work_size.ndims) );
    const cl_int err =
        clEnqueueNDRangeKernel( scl_dev->queue,
                                kernel.kernel,
                                global_work_size.ndims,
                                NULL, /* no offset */
                                global_work_size.sizes,
                                (local_work_size.ndims < 1 ? NULL : local_work_size.sizes),
                                0, /* no events in wait list */
                                NULL, /* empty wait list */
                                NULL );
    sclCheckError(err, "clEnqueueNDRangeKernel error in sclEnqueueKernel for kernel %s: %s\n", kernel.kernel_name, sclGetErrorString(err));
    nkernels++;
    if (nkernels >= SYNC_EVERY) {
        sclDeviceSynchronize();
        nkernels = 0;
    }
}

void sclSetKernelArg( sclKernel kernel, int argnum, size_t typeSize, void *argument )
{
    sclCheckDeviceInitialized();

    const cl_int err = clSetKernelArg( kernel.kernel, argnum, typeSize, argument );
    sclCheckError(err, "clSetKernelArg error in sclSetKernelArg number %d for kernel %s: %s\n", argnum, kernel.kernel_name, sclGetErrorString(err));
}

static void _sclVSetKernelArgs( sclKernel kernel, const char *fmt, va_list argList )
{
    int arg_count = 0;
    void* argument;
    size_t actual_size;
    int int_arg;
    long long_arg;
    float float_arg;
    cl_mem mem_arg;

    /* bail out if no format string is given */
    if (fmt == NULL)
        return;

    for( const char *p = fmt; *p != '\0'; p++ ) {
        if ( *p == SCL_SEPARATOR ) {
            switch( *++p ) {
            case SCL_VALUE:
                actual_size = va_arg( argList, size_t );
                argument = va_arg( argList, void* );
                sclSetKernelArg(kernel, arg_count, actual_size, argument);
                arg_count++;
                break;
            case SCL_BUFFER:
                mem_arg = va_arg(argList, cl_mem);
                sclSetKernelArg(kernel, arg_count, sizeof(cl_mem), (void*)&mem_arg);
                arg_count++;
                break;
            case SCL_LOCALMEM:
                actual_size = va_arg( argList, size_t );
                sclSetKernelArg(kernel, arg_count, actual_size, NULL);
                arg_count++;
                break;
            case SCL_INT:
                int_arg = va_arg( argList, int );
                sclSetKernelArg(kernel, arg_count, sizeof(int), &int_arg);
                arg_count++;
                break;
            case SCL_LONG:
                long_arg = va_arg( argList, long );
                sclSetKernelArg(kernel, arg_count, sizeof(long), &long_arg);
                arg_count++;
                break;
            case SCL_FLOAT:
                float_arg = (float)va_arg( argList, double ); /* floats are promoted to double when passed through ... */
                sclSetKernelArg(kernel, arg_count, sizeof(float), &float_arg);
                arg_count++;
                break;
            default:
                sclPanic("Unrecognized character '%c' in format string \"%s\" for kernel %s\n", *p, fmt, kernel.kernel_name);
            }
        } else if (*p != ' ') {
            sclPanic("Unrecognized character '%c' in format string \"%s\" for kernel %s\n", *p, fmt, kernel.kernel_name);
        }
    }
}

void sclSetKernelArgs( sclKernel kernel, const char *fmt, ... )
{
    va_list argList;
    va_start( argList, fmt );
    _sclVSetKernelArgs( kernel, fmt, argList );
    va_end( argList );
}

/***

### Interface

```C
void sclSetArgsLaunchKernel( sclKernel kernel,
                             const sclDim global_work_size,
                             const sclDim local_work_size,
                             const char *fmt, ... )
```

### Parameters

- `kernel`: kernel to execute, created by `sclCreateKernel()`

- `global_work_size`: problem size

- `local_work_size`: workgroup size

- `fmt`: format string for the parameters (see below)

### Synopsis

Format      Meaning
---------   ---------------------------------------------------
`:b`        The corresponding parameter must be of type `cl_mem`
            that represents a memory buffer to be passed to the kernel.
            The corresponding kernel parameter must be of type
            `T *` (pointer to some type `T`).

`:L`        The corresponding parameter must be of type `size_t`,
            and represents the size of a local memory block that
            is allocated and passed as a parameter of type `__local`
            or `local` to the kernel call.

`:v`        The corresponding parameter must be a pair (`size_t`, `void *`),
            representing a memory block whose content is passed to the
            appropriate formal parameter to the kernel. This can be
            useful to pass complex data types (e.g., structures) to
            kernels.

`:d`        The corresponding parameter must be of type `int`,
            and corresponds to an `int` kernel parameter.

`:l`        The corresponding parameter must be of type `long`,
            and corresponds to a `long` kernel parameter.

`:f`        The corresponding parameter must be of type `float`,
            and corresponds to a `float` kernel parameter.
---------   ---------------------------------------------------

`global_work_size` and `local_work_size` must have the same numeber of
dimensions (e.g., they both must be created using `DIM1()` or `DIM2()`
or `DIM3()`); furthermore, the number of work-items along each
dimension of `global_work_size` must be an integer multiple of the
corresponding number in `local_work_size`.

For example, `global_work_size` set to `DIM3(128, 128, 3)` is
compatible with `local_work_size` set to `DIM3(64, 32, 1)`, but is
_not_ compatible with `local_work_size` set to `DIM3(112, 128, 3)`
since 128 is not an integer multiple of 112.

### Example

 ***/
void sclSetArgsLaunchKernel( sclKernel kernel,
                             const sclDim global_work_size,
                             const sclDim local_work_size,
                             const char *fmt, ... )
{
    va_list argList;
    va_start( argList, fmt );
    _sclVSetKernelArgs( kernel, fmt, argList );
    va_end( argList );
    sclLaunchKernel( kernel, global_work_size, local_work_size );
}

/***

```C
void sclSetArgsEnqueueKernel( sclKernel kernel,
                              const sclDim global_work_size,
                              const sclDim local_work_size,
                              const char *fmt, ... )
```

Enqueue a kernel execution. Parameters are the same as
`sclSetArgsLaunchKernel()`

> **Note.** Enqueueing a large number of kernels in rapid succession
> might fill up the command queue and cause an error. To prevent this,
> `simpleCL` periodically inserts a `sclDeviceSynchronize()` command
> after asynchronous kernel launches.

 ***/
void sclSetArgsEnqueueKernel( sclKernel kernel,
                              const sclDim global_work_size,
                              const sclDim local_work_size,
                              const char *fmt, ... )
{
    va_list argList;
    va_start( argList, fmt );
    _sclVSetKernelArgs( kernel, fmt, argList );
    va_end( argList );
    sclEnqueueKernel( kernel, global_work_size, local_work_size );
}



/******************************************************************************
 **
 ** DEVICE MANAGEMENT
 **
 ******************************************************************************/

void sclPrintHardwareStatus( void )
{
    char platform[MAX_STR_LEN];
    cl_bool deviceAV;
    cl_int err;

    sclCheckDeviceInitialized();
    err = clGetPlatformInfo( scl_dev->platform_id,
                             CL_PLATFORM_NAME,
                             sizeof(platform),
                             platform,
                             NULL );
    if ( err == CL_SUCCESS ) {
        fprintf(stderr, "Platform object alive\n");
    } else {
        fprintf(stderr, "%s\n", sclGetErrorString(err));
    }

    err = clGetDeviceInfo( scl_dev->device_id,
                           CL_DEVICE_AVAILABLE,
                           sizeof(cl_bool),
                           (void*)(&deviceAV),
                           NULL );
    if ( err == CL_SUCCESS && deviceAV ) {
        fprintf(stderr, "Device object alive and device available\n");
    } else if ( err == CL_SUCCESS ) {
        fprintf(stderr, "Device object alive and device NOT available\n");
    } else {
        fprintf(stderr, "Device object not alive\n");
    }
}

/***

Wait for completion of all pending commands.

 ***/
cl_int sclDeviceSynchronize( void )
{
    sclCheckDeviceInitialized();

    const cl_int err = clFinish( scl_dev->queue );
    sclCheckError(err, "clFinish error in sclDeviceSynchronize: %s\n", sclGetErrorString(err));
    return err;
}

/***

Return the lowest integer multiple of `m` that is greater than or
equal to `s`. For example, if `s=13, m=5` this function returns `15`,
which is the lowest multiple of `5` that is greater than or equal to
`13`. If `s=18, m=6` this function returns `18`, which is already an
integer multiple of `6`.

The function is useful when computing the size of a global grid as the
lowest multiple of the default local dimensions that has enough items
as the input.

 ***/
size_t sclRoundUp(size_t s, size_t m)
{
    return ((s+m-1)/m)*m;
}

void sclWGSetup1D(size_t xsize, sclDim *grid, sclDim *block)
{
    *block = DIM1(SCL_DEFAULT_WG_SIZE);
    *grid = DIM1(sclRoundUp(xsize, SCL_DEFAULT_WG_SIZE));
}

void sclWGSetup2D(size_t xsize, size_t ysize, sclDim *grid, sclDim *block)
{
    *block = DIM2(SCL_DEFAULT_WG_SIZE2D, SCL_DEFAULT_WG_SIZE2D);
    *grid = DIM2(sclRoundUp(xsize, SCL_DEFAULT_WG_SIZE2D),
                 sclRoundUp(ysize, SCL_DEFAULT_WG_SIZE2D));
}

void sclWGSetup3D(size_t xsize, size_t ysize, size_t zsize, sclDim *grid, sclDim *block)
{
    *block = DIM3(SCL_DEFAULT_WG_SIZE3D, SCL_DEFAULT_WG_SIZE3D, SCL_DEFAULT_WG_SIZE3D);
    *grid = DIM3(sclRoundUp(xsize, SCL_DEFAULT_WG_SIZE3D),
                 sclRoundUp(ysize, SCL_DEFAULT_WG_SIZE3D),
                 sclRoundUp(zsize, SCL_DEFAULT_WG_SIZE3D));
}
