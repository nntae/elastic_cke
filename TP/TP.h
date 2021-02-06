#include <cooperative_groups.h>

namespace cg = cooperative_groups;
// Utilities and system includes
#include <helper_string.h>    // helper for string parsing
#include <helper_image.h>     // helper for image and data comparison
#include <helper_cuda.h>      // helper for cuda error checking functions

#define FLOOR(a,b) (a-(a%b))

typedef struct {
    const char *sSDKsample = "Transpose";

    int tile_dim, block_rows, mul_factor, max_tiles;
    int matrix_size_x, matrix_size_y, size_x, size_y;
    float *h_idata, *h_odata, *transposeGold, *gold;
    float *d_idata, *d_odata;
    int num_reps;
    size_t mem_size;
    bool success;
    int *zc_slc;
} t_TP_params;

int launch_orig_TP(void *kstub);
int launch_slc_TP(void *kstub);

int TP_start_kernel(void *arg);
int TP_end_kernel(void *arg);

int TP_start_mallocs(void *arg);
int TP_start_transfers(void *arg);