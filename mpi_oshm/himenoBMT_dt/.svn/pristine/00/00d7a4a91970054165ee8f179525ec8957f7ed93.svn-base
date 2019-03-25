#ifndef _CUDABMTKERNEL_CUH
#define _CUDABMTKERNEL_CUH

#include "commonBMT.h"
#include "cuda_runtime_api.h"

cudaError_t bmtInitDeviceMemory(
    Matrix * fa_h,   Matrix * fb_h,    Matrix * fc_h,
    Matrix * fp_h,   Matrix * fwrk1_h, Matrix * fwrk2_h,
    Matrix * fbnd_h, int      peid);

cudaError_t bmtCudaJacobi(PRECISION * gosa, Matrix * fp_h,
    int imax, int jmax, int kamx);

#endif //_CUDABMTKERNEL_CUH

