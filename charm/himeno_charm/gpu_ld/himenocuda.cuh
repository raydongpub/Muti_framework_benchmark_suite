#ifndef _CUDABMTKERNEL_CUH
#define _CUDABMTKERNEL_CUH

#include "commonBMT.h"
#include "cuda_runtime_api.h"

cudaError_t bmtInitDeviceMemory(
    Matrix * fa_h,   Matrix * fb_h,    Matrix * fc_h,
    Matrix * fp_h,   Matrix * fwrk1_h, Matrix * fwrk2_h,
    Matrix * fbnd_h, int peid, BMT_Config  config, bool copy
/*    PRECISION ** fa_d, PRECISION ** fb_d, PRECISION ** fc_d, PRECISION ** fp_d, PRECISION ** fwrk1_d,
    PRECISION ** fwrk2_d, PRECISION ** fbnd_d, PRECISION ***** a_d, PRECISION ***** b_d, PRECISION ***** c_d,
    PRECISION **** p_d, PRECISION **** wrk1_d, PRECISION **** wrk2_d, PRECISION **** bnd_d,
    PRECISION ** gosa_d, PRECISION ***** a_h, PRECISION ***** b_h, PRECISION ***** c_h,
    PRECISION **** p_h, PRECISION **** wrk1_h, PRECISION **** wrk2_h, PRECISION **** bnd_h,
    PRECISION ** gosa_h*/);

cudaError_t bmtCudaJacobi(PRECISION * gosa, Matrix * fp_h,
    int imax, int jmax, int kamx, BMT_Config config, int peid
/*    PRECISION * fa_d, PRECISION * fb_d, PRECISION * fc_d, PRECISION * fp_d, PRECISION * fwrk1_d,
    PRECISION* fwrk2_d, PRECISION* fbnd_d, PRECISION **** a_d, PRECISION **** b_d, PRECISION **** c_d,
    PRECISION*** p_d, PRECISION *** wrk1_d, PRECISION *** wrk2_d, PRECISION *** bnd_d,
    PRECISION * gosa_d, PRECISION **** a_h, PRECISION **** b_h, PRECISION **** c_h,
    PRECISION *** p_h, PRECISION *** wrk1_h, PRECISION *** wrk2_h, PRECISION *** bnd_h,
    PRECISION * gosa_h*/);

#endif //_CUDABMTKERNEL_CUH

