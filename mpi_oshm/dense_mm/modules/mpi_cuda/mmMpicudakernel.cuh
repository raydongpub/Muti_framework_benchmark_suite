#ifndef _MMMPICUDAKERNEL_CUH
#define _MMMPICUDAKERNEL_CUH

#include "cuda_runtime_api.h"
#include "matdataset.h"
#include "main.h"

namespace GPU_MatrixDataset {
    typedef struct {
        int cstart;
        int cnum;
        int peidx;
        int numpe;
        int width;
        int height;
        int vec_size;
        int result_size;
        int numcom;
    } GPUConfig;

    typedef struct {
        PRECISION * elements;
   } MatrixList;
};

cudaError_t CudaInitialize(PRECISION * veca, PRECISION * vecb, PRECISION * vecc, parConfig * par, int rank, int numpes);
cudaError_t CudaClean();
cudaError_t ComputeMatrixAttributes(int num_blocks, int num_threads);

#endif //_MMOSHMCUDAKERNEL_CUH

