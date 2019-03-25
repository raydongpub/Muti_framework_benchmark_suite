#include "cuda_runtime_api.h"
#include "nbody_main.h"
#include "nnOshmCudaKernel.cuh"
#include "ParticleDataset.h"
#include "mpi.h"
#include <math.h>
#include <string.h>
#include <iostream>
#include <shmem.h>

#ifndef _DOUBLE_PRECISION
#define MPI_PRECISION MPI_FLOAT
#else //_DOUBLE_PRECISION
#define MPI_PRECISION MPI_DOUBLE
#endif //_DOUBLE_PRECISION

#define MPI_NN_ROOT 0

using namespace std;

//General
NbodyConfig               * plConfig;
ParticleDataset           * plDataset;
ParticleDataset           * shDataset;
PRECISION                   sec = 0.0;
int                       * numParticles;

//CUDA
GPU_ParticleDataset::GPUConfig gConfig;

//MPI
int               rank;

int LibSetup(NbodyConfig * config,
        ParticleDataset * dataset) {

    plConfig  = config;
    plDataset = dataset;

    return 0;
}

inline void * CollectiveInitialize(int * localC, int * localD) {
    bool osh = true;
    ParticleDataset::Particle * localB;
    int mid = _my_pe();
    numParticles = (int *)shmalloc(sizeof(int));
    //Sync-up the number of total particles.
    if (!mid) {
        *numParticles             = plDataset->mNumParticles;
    }

    shmem_barrier_all();
    if (mid) {
        shmem_int_get(numParticles, numParticles, sizeof(int), 0);
    }
    shmem_barrier_all();

    shDataset = new ParticleDataset(*numParticles, osh);

    if (!mid) {
        memcpy(shDataset->mpParticle, plDataset->mpParticle,
            *numParticles * sizeof(ParticleDataset::Particle));
    }

    shmem_barrier_all();
    if (mid) {
        shmem_getmem(shDataset->mpParticle, shDataset->mpParticle, *numParticles * sizeof(ParticleDataset::Particle), 0);
    }

  //Determine number of particles handled locally.
    int divCnt = *numParticles / plConfig->mParams.commSize;
    int remCnt = *numParticles % plConfig->mParams.commSize;
    int total_pe = _num_pes();
    int idx = _my_pe();
    //Adjust local particle-cpimnts and displacements.
    int * cntList, * dispList;
    cntList    = new int [plConfig->mParams.commSize];
    dispList   = new int [plConfig->mParams.commSize];
    int disp = 0;
    for (int i=0;i<plConfig->mParams.commSize;i++) {
        int addition   = ((i + 1) <= remCnt) ? 1 : 0;
        cntList[i]   = divCnt + addition;
        dispList[i]  = disp;
        disp          += divCnt + addition;
    }
    *localC = cntList[plConfig->mParams.rank];
    *localD = dispList[plConfig->mParams.rank];

    delete [] cntList;
    delete [] dispList;
    localB  = new ParticleDataset::Particle[*localC];

    printf("[%d]: %d/%d\n", plConfig->mParams.rank, localD, localC);
    return localB;
}

inline void CollectiveClean(ParticleDataset::Particle *localB) {
   delete [] localB;
}

void LibEntry(int argc, char **argv, int num_blocks, int num_threads) {
#define SET_PARTICLE(str1, str2)        \
    x##str1##Pos = localBuf[str2].xPos; \
    y##str1##Pos = localBuf[str2].yPos; \
    z##str1##Pos = localBuf[str2].zPos; \
    x##str1##Vel = localBuf[str2].xVel; \
    y##str1##Vel = localBuf[str2].yVel; \
    z##str1##Vel = localBuf[str2].zVel; \
    x##str1##Acc = localBuf[str2].xAcc; \
    y##str1##Acc = localBuf[str2].yAcc; \
    z##str1##Acc = localBuf[str2].zAcc; \
    mass##str1 = localBuf[str2].mass
#define SET_IPARTICLE(str1, str2)                   \
    x##str1##Pos = shDataset->mpParticle[str2].xPos; \
    y##str1##Pos = shDataset->mpParticle[str2].yPos; \
    z##str1##Pos = shDataset->mpParticle[str2].zPos; \
    mass##str1 = shDataset->mpParticle[str2].mass

#define CHK_ERR(str)                                             \
    do {                                                         \
        cudaError_t ce = str;                                    \
        if (ce != cudaSuccess) {                                 \
            cout << "Error: " << cudaGetErrorString(ce) << endl; \
        }                                                        \
    } while (0)

    ParticleDataset::Particle * localBuf;
    int                         localCnt;
    int                         localDisp;
    int                         idx;

    localBuf = (ParticleDataset::Particle *)CollectiveInitialize(&localCnt, &localDisp);

    PRECISION step         = plConfig->mParams.timeRes;
    PRECISION duration     = plConfig->mParams.duration;
    PRECISION grav         = plConfig->mParams.gravConstant;
    idx = _my_pe();

    gConfig.localCnt  = localCnt;
    gConfig.localDisp = localDisp;
    CHK_ERR(    CudaInitialize(shDataset, localBuf, plConfig, gConfig));

    cout << "[" << plConfig->mParams.rank << "/"
       << plConfig->mParams.commSize << "]: "
       << localCnt << endl;
    
    for (sec=0.0;sec<duration;sec+=step) {
        CHK_ERR(    ComputeParticleAttributes(num_blocks, num_threads));
        shmem_barrier_all();
        shmem_putmem(shDataset->mpParticle + localDisp, localBuf,
            localCnt * sizeof(ParticleDataset::Particle), 0);
        if (idx) {
            shmem_getmem(shDataset->mpParticle, shDataset->mpParticle,
                *numParticles * sizeof(ParticleDataset::Particle), 0);
        }
        shmem_barrier_all();

        if (!plConfig->mParams.rank)
            cout << "secs: " << sec << "/" << duration << "\xd";
    }

    CHK_ERR(    CudaClean());
    CollectiveClean(localBuf);
}

int LibCleanUp(void) {

    if (!_my_pe())
        shDataset->SaveToFile("oshm_cuda.bin");
    shfree(numParticles);

    return 0;
}

