#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <float.h> // for FLT_MAX

#include "CudaMath.h"
#include "dds.h"
#include "permutations.h"

// Definitions
#define INPUT_IMAGE "lena_std.ppm"
#define REFERENCE_IMAGE "lena_ref.dds"

#define ERROR_THRESHOLD 0.02f

#define NUM_THREADS 256

#define __debugsync()

typedef struct {
    char *sSDKsample;
    bool printInfo;

    char *image_path;
    uint compressedSize, memSize;
    uint w, h;
    uint permutations[1024];
    uint *d_data, *d_result, *d_permutations, *h_result;
    uint *block_image;
    int *zc_slc;
} t_DXTC_params;

int launch_orig_DXTC(void *kstub);
int launch_slc_DXTC(void *kstub);

int DXTC_start_kernel(void *arg);
int DXTC_end_kernel(void *arg);

int DXTC_start_mallocs(void *arg);
int DXTC_start_transfers(void *arg);