#include "cuda_runtime_api.h"
#include "nbody_main.h"
#include "nnMpiCudaKernel.cuh"
#include "ParticleDataset.h"
#include "mpi.h"
#include <math.h>
#include <string.h>
#include <iostream>

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
int                       * cntList;
int                       * dispList;
ParticleDataset::Particle * localBuf;
int                         localCnt;
int                         localDisp;
PRECISION                   sec = 0.0;

//CUDA
GPU_ParticleDataset::GPUConfig gConfig;

//MPI
int               rank;
MPI_Datatype      mpiParticleType;
int               structCount;
int             * structBlock;
MPI_Aint        * structDisplacement;
MPI_Datatype    * structDatatype;

int LibSetup(NbodyConfig * config,
        ParticleDataset * dataset) {

    plConfig  = config;
    plDataset = dataset;

    return 0;
}

inline void DefineDatatype() {

    structCount = sizeof(ParticleDataset::Particle)/sizeof(PRECISION);

    structBlock = new int[structCount];
    for (int idx=0;idx<structCount;idx++)
        structBlock[idx] = 1;

    structDisplacement = new MPI_Aint[structCount];
    structDisplacement[0] = 0;
    for (int idx=1;idx<structCount;idx++) {
        structDisplacement[idx] = structDisplacement[idx - 1] +
            sizeof(PRECISION);
    }

    structDatatype = new MPI_Datatype[structCount];
    for (int idx=0;idx<structCount;idx++)
        structDatatype[idx] = MPI_PRECISION;

    MPI_Type_struct(structCount, structBlock, structDisplacement,
        structDatatype, &mpiParticleType);
    MPI_Type_commit(&mpiParticleType);
}

inline void FreeDatatype() {

    MPI_Type_free(&mpiParticleType);

    delete [] structBlock;
    delete [] structDisplacement;
    delete [] structDatatype;
}

inline void CollectiveInitialize() {

    DefineDatatype();

    //Sync-up the number of total particles.
    int numParticles;
    bool oshm = false;
    if (plConfig->mParams.rank == MPI_NN_ROOT)
        numParticles = plDataset->mNumParticles;

    MPI_Bcast(&numParticles, 1, MPI_INT, MPI_NN_ROOT, MPI_COMM_WORLD);

    if (plConfig->mParams.rank != MPI_NN_ROOT)
        plDataset = new ParticleDataset(numParticles, oshm);

    MPI_Bcast(plDataset->mpParticle, plDataset->mNumParticles,
        mpiParticleType, MPI_NN_ROOT, MPI_COMM_WORLD);

    //Determine number of particles handled locally.
    int divCnt = numParticles / plConfig->mParams.commSize;
    int remCnt = numParticles % plConfig->mParams.commSize;

    //Construct lists of particle-counts and displacements.
    cntList    = new int [plConfig->mParams.commSize];
    dispList   = new int [plConfig->mParams.commSize];
#if 0
    if (remCnt) {
        if (!plConfig->mParams.rank)
           cerr << "Error: input cannot be uniformly divided." << endl;
        MPI_Abort(MPI_COMM_WORLD, 255);
    }
    int disp = 0;
    for (int idx=0;idx<plConfig->mParams.commSize;idx++) {
        cntList[idx]   = divCnt;
        dispList[idx]  = disp;
        disp          += divCnt;
    }
#else
/*
    if ((plConfig->mParams.rank+1) <= remCnt)
        localCnt = divCnt + 1;
    else
        localCnt = divCnt;
*/
    int disp = 0;
    for (int idx=0;idx<plConfig->mParams.commSize;idx++) {
        int addition   = ((idx + 1) <= remCnt) ? 1 : 0;
        cntList[idx]   = divCnt + addition;
        dispList[idx]  = disp;
        disp          += divCnt + addition;
    }

#endif

    //Adjust local particle-cpimnts and displacements.
    localCnt  = cntList[plConfig->mParams.rank];
    localDisp = dispList[plConfig->mParams.rank];
    localBuf  = new ParticleDataset::Particle[localCnt];
}

inline void CollectiveClean() {

    delete [] dispList;
    delete [] cntList;
    delete [] localBuf;

    if (plConfig->mParams.rank != MPI_NN_ROOT)
        delete plDataset;

    FreeDatatype();
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
    x##str1##Pos = plDataset->mpParticle[str2].xPos; \
    y##str1##Pos = plDataset->mpParticle[str2].yPos; \
    z##str1##Pos = plDataset->mpParticle[str2].zPos; \
    mass##str1 = plDataset->mpParticle[str2].mass

#define CHK_ERR(str)                                             \
    do {                                                         \
        cudaError_t ce = str;                                    \
        if (ce != cudaSuccess) {                                 \
            cout << "Error: " << cudaGetErrorString(ce) << endl; \
        }                                                        \
    } while (0)

    CollectiveInitialize();

    gConfig.localCnt  = localCnt;
    gConfig.localDisp = localDisp;
    CHK_ERR(    CudaInitialize(plDataset, localBuf, plConfig, gConfig));

    cout << "[" << plConfig->mParams.rank << "/"
       << plConfig->mParams.commSize << "]: "
       << localCnt << endl;

    PRECISION step     = plConfig->mParams.timeRes;
    PRECISION duration = plConfig->mParams.duration;

    for (sec=0.0;sec<duration;sec+=step) {
        CHK_ERR(    ComputeParticleAttributes(num_blocks, num_threads));

        MPI_Allgatherv(localBuf, localCnt, mpiParticleType,
            plDataset->mpParticle, cntList, dispList, mpiParticleType,
            MPI_COMM_WORLD);

        if (!plConfig->mParams.rank)
            cout << "secs: " << sec << "/" << duration << "\xd";
    }

    CHK_ERR(    CudaClean());
    CollectiveClean();
}

int LibCleanUp(void) {

    if (!plConfig->mParams.rank)
        plDataset->SaveToFile("mpi_cuda.bin");

    return 0;
}

