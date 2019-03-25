#ifndef _NNOSHMCUDAKERNEL_CUH
#define _NNOSHMCUDAKERNEL_CUH

#include "cuda_runtime_api.h"
#include "ParticleDataset.h"

namespace GPU_ParticleDataset {
    typedef struct {
        int localCnt;
        int localDisp;
        int pid;
        int dev;
        PRECISION step;
        PRECISION grav;
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

cudaError_t CudaInitialize(ParticleDataset * dataset,
    ParticleDataset::Particle * locBuf,
    GPU_ParticleDataset::GPUConfig gConfig);
cudaError_t CudaClean();
cudaError_t ComputeParticleAttributes(ParticleDataset * dataset,
    ParticleDataset::Particle * locBuf,
    GPU_ParticleDataset::GPUConfig gConfig, PRECISION step, PRECISION grav, int peid, int pid, int localCnt, int localDisp, int numParticles);

#endif //_NNOSHMCUDAKERNEL_CUH

