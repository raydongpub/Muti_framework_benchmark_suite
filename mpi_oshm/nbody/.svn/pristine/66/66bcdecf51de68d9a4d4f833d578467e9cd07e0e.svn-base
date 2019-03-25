#ifndef _NNIVMCUDAKERNEL_CUH
#define _NNIVMCUDAKERNEL_CUH

#include "cuda_runtime_api.h"
#include "ParticleDataset.h"
#include "NbodyConfig.h"
#include <iostream>

namespace GPU_ParticleDataset {
    typedef struct {
        int localCnt;
        int localDisp;
    } GPUConfig;

    typedef struct {
        PRECISION * xPos;
        PRECISION * yPos;
        PRECISION * zPos;
        PRECISION * xVel;
        PRECISION * yVel;
        PRECISION * zVel;
        PRECISION * xAcc;
        PRECISION * yAcc;
        PRECISION * zAcc;
        PRECISION * mass;
    } ParticleList;
};

inline void CHK_ERR(cudaError_t ce) {
    
    if (ce != cudaSuccess) {
        cout << "CUDA_ERROR: " << cudaGetErrorString(ce) << endl;
        exit(0);
    }
}

cudaError_t CudaInitialize(ParticleDataset * dataset,
    ParticleDataset::Particle * locBuf, NbodyConfig * config,
    GPU_ParticleDataset::GPUConfig gConfig);
cudaError_t CudaClean();
cudaError_t ComputeParticleAttributes();

#endif //_NNIVMCUDAKERNEL_CUH

