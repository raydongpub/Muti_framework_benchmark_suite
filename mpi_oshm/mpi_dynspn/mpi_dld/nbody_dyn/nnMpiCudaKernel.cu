#include "nnMpiCudaKernel.cuh"
using namespace GPU_ParticleDataset;

#define _DEBUG
#ifdef _DEBUG
#include <iostream>
using namespace std;
#endif //__DEBUG

ParticleDataset           * hpDataset;
ParticleDataset::Particle * hpLocBuf;
NbodyConfig               * hpConfig;
GPUConfig                   gpuConfig;

ParticleList gParList_h;
ParticleList gParList_d;
ParticleList lParList_h;
ParticleList lParList_d;

__global__ void ComputeParticleAttributes_Kernel(
    PRECISION * gxPos, PRECISION * gyPos, PRECISION * gzPos,
    PRECISION * gxVel, PRECISION * gyVel, PRECISION * gzVel,
    PRECISION * gxAcc, PRECISION * gyAcc, PRECISION * gzAcc,
    PRECISION * gmass,
    PRECISION * lxPos, PRECISION * lyPos, PRECISION * lzPos,
    PRECISION * lxVel, PRECISION * lyVel, PRECISION * lzVel,
    PRECISION * lxAcc, PRECISION * lyAcc, PRECISION * lzAcc,
    PRECISION * lmass,
    PRECISION   step,  PRECISION grav,
    int     localCnt,  int  localDisp,
    int totalCnt) {
#if 1
    int gOffset = ((blockIdx.z * (gridDim.x * gridDim.y)) +
                   (blockIdx.y * gridDim.x) + blockIdx.x) *
                   (blockDim.z * blockDim.y * blockDim.x);

    int gid     = ((threadIdx.z * (blockDim.x * blockDim.y)) +
                   (threadIdx.y * blockDim.x) + threadIdx.x) +
                    gOffset;

    int total   = (gridDim.x * gridDim.y * gridDim.z) *
                  (blockDim.x * blockDim.y * blockDim.z);
#else
    int gid     = (blockIdx.x * blockDim.x) + threadIdx.x;
    int total   = gridDim.x * blockDim.x;
#endif

    int stride  = (totalCnt / total) + 1;

    for (int iIdx=0;iIdx<stride;iIdx++) {

        int pid = (iIdx * total) + gid;
        if (pid >= localCnt)
            break;

        PRECISION radius_s, radius;
        PRECISION force, force_x = 0.0, force_y = 0.0, force_z = 0.0;
        PRECISION x1Pos = lxPos[pid], y1Pos = lyPos[pid], z1Pos = lzPos[pid];
        PRECISION mass1 = lmass[pid];
#if 1
        for (int jIdx=0;jIdx<totalCnt;jIdx++) {
            if (jIdx != (localDisp + pid)) {
                PRECISION x2Pos = gxPos[jIdx];
                PRECISION y2Pos = gyPos[jIdx];
                PRECISION z2Pos = gzPos[jIdx];
                PRECISION mass2 = gmass[jIdx];

                radius_s = ((x2Pos - x1Pos) * (x2Pos - x1Pos)) +
                           ((y2Pos - y1Pos) * (y2Pos - y1Pos)) +
                           ((z2Pos - z1Pos) * (z2Pos - z1Pos));
                radius   = sqrt(radius_s);
                force    = (grav * mass1 * mass2) / radius_s;
                force_x += force * ((x2Pos - x1Pos) / radius);
                force_y += force * ((y2Pos - y1Pos) / radius);
                force_z += force * ((z2Pos - z1Pos) / radius);
            }
        }
#endif
        lxAcc[pid]  = force_x / mass1;
        lyAcc[pid]  = force_y / mass1;
        lzAcc[pid]  = force_z / mass1;
        lxVel[pid] += lxAcc[pid] * step;
        lyVel[pid] += lyAcc[pid] * step;
        lzVel[pid] += lzAcc[pid] * step;
        lxPos[pid] += lxVel[pid] * step;
        lyPos[pid] += lyVel[pid] * step;
        lzPos[pid] += lzVel[pid] * step;
    }
}

#define CHK_ERR(str)           \
    do {                       \
        cudaError_t ce = str;  \
        if (ce != cudaSuccess) \
            return ce;         \
    } while (0)

cudaError_t ConvertHostToDevice() {

    for (int idx=0;idx<hpDataset->mNumParticles;idx++) {
        gParList_h.xPos[idx] = hpDataset->mpParticle[idx].xPos;
        gParList_h.yPos[idx] = hpDataset->mpParticle[idx].yPos;
        gParList_h.zPos[idx] = hpDataset->mpParticle[idx].zPos;
        gParList_h.xVel[idx] = hpDataset->mpParticle[idx].xVel;
        gParList_h.yVel[idx] = hpDataset->mpParticle[idx].yVel;
        gParList_h.zVel[idx] = hpDataset->mpParticle[idx].zVel;
        gParList_h.xAcc[idx] = hpDataset->mpParticle[idx].xAcc;
        gParList_h.yAcc[idx] = hpDataset->mpParticle[idx].yAcc;
        gParList_h.zAcc[idx] = hpDataset->mpParticle[idx].zAcc;
        gParList_h.mass[idx] = hpDataset->mpParticle[idx].mass;
    }

    memcpy(lParList_h.xPos, gParList_h.xPos + gpuConfig.localDisp,
        gpuConfig.localCnt * sizeof(PRECISION));
    memcpy(lParList_h.yPos, gParList_h.yPos + gpuConfig.localDisp,
        gpuConfig.localCnt * sizeof(PRECISION));
    memcpy(lParList_h.zPos, gParList_h.zPos + gpuConfig.localDisp,
        gpuConfig.localCnt * sizeof(PRECISION));
    memcpy(lParList_h.xVel, gParList_h.xVel + gpuConfig.localDisp,
        gpuConfig.localCnt * sizeof(PRECISION));
    memcpy(lParList_h.yVel, gParList_h.yVel + gpuConfig.localDisp,
        gpuConfig.localCnt * sizeof(PRECISION));
    memcpy(lParList_h.zVel, gParList_h.zVel + gpuConfig.localDisp,
        gpuConfig.localCnt * sizeof(PRECISION));
    memcpy(lParList_h.xAcc, gParList_h.xAcc + gpuConfig.localDisp,
        gpuConfig.localCnt * sizeof(PRECISION));
    memcpy(lParList_h.yAcc, gParList_h.yAcc + gpuConfig.localDisp,
        gpuConfig.localCnt * sizeof(PRECISION));
    memcpy(lParList_h.zAcc, gParList_h.zAcc + gpuConfig.localDisp,
        gpuConfig.localCnt * sizeof(PRECISION));
    memcpy(lParList_h.mass, gParList_h.mass + gpuConfig.localDisp,
        gpuConfig.localCnt * sizeof(PRECISION));
#if 1
    CHK_ERR(    cudaMemcpy(gParList_d.xPos, gParList_h.xPos,
        hpDataset->mNumParticles * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(gParList_d.yPos, gParList_h.yPos,
        hpDataset->mNumParticles * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(gParList_d.zPos, gParList_h.zPos,
        hpDataset->mNumParticles * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(gParList_d.xVel, gParList_h.xVel,
        hpDataset->mNumParticles * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(gParList_d.yVel, gParList_h.yVel,
        hpDataset->mNumParticles * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(gParList_d.zVel, gParList_h.zVel,
        hpDataset->mNumParticles * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(gParList_d.xAcc, gParList_h.xAcc,
        hpDataset->mNumParticles * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(gParList_d.yAcc, gParList_h.yAcc,
        hpDataset->mNumParticles * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(gParList_d.zAcc, gParList_h.zAcc,
        hpDataset->mNumParticles * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(gParList_d.mass, gParList_h.mass,
        hpDataset->mNumParticles * sizeof(PRECISION), cudaMemcpyHostToDevice));
#endif
    CHK_ERR(    cudaMemcpy(lParList_d.xPos, lParList_h.xPos,
       gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(lParList_d.yPos, lParList_h.yPos,
       gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(lParList_d.zPos, lParList_h.zPos,
       gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(lParList_d.xVel, lParList_h.xVel,
       gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(lParList_d.yVel, lParList_h.yVel,
       gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(lParList_d.zVel, lParList_h.zVel,
       gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(lParList_d.xAcc, lParList_h.xAcc,
       gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(lParList_d.yAcc, lParList_h.yAcc,
       gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(lParList_d.zAcc, lParList_h.zAcc,
       gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(lParList_d.mass, lParList_h.mass,
       gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyHostToDevice));

    return cudaSuccess;
}

cudaError_t ConvertDeviceToHost() {

    CHK_ERR(    cudaMemcpy(lParList_h.xPos, lParList_d.xPos,
        gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyDeviceToHost));
    CHK_ERR(    cudaMemcpy(lParList_h.yPos, lParList_d.yPos,
        gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyDeviceToHost));
    CHK_ERR(    cudaMemcpy(lParList_h.zPos, lParList_d.zPos,
        gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyDeviceToHost));
    CHK_ERR(    cudaMemcpy(lParList_h.xVel, lParList_d.xVel,
        gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyDeviceToHost));
    CHK_ERR(    cudaMemcpy(lParList_h.yVel, lParList_d.yVel,
        gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyDeviceToHost));
    CHK_ERR(    cudaMemcpy(lParList_h.zVel, lParList_d.zVel,
        gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyDeviceToHost));
    CHK_ERR(    cudaMemcpy(lParList_h.xAcc, lParList_d.xAcc,
        gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyDeviceToHost));
    CHK_ERR(    cudaMemcpy(lParList_h.yAcc, lParList_d.yAcc,
        gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyDeviceToHost));
    CHK_ERR(    cudaMemcpy(lParList_h.zAcc, lParList_d.zAcc,
        gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyDeviceToHost));
    CHK_ERR(    cudaMemcpy(lParList_h.mass, lParList_d.mass,
        gpuConfig.localCnt * sizeof(PRECISION), cudaMemcpyDeviceToHost));

    for (int idx=0;idx<gpuConfig.localCnt;idx++) {
	hpLocBuf[idx].xPos = lParList_h.xPos[idx];
        hpLocBuf[idx].yPos = lParList_h.yPos[idx];
        hpLocBuf[idx].zPos = lParList_h.zPos[idx];
	hpLocBuf[idx].xVel = lParList_h.xVel[idx];
        hpLocBuf[idx].yVel = lParList_h.yVel[idx];
        hpLocBuf[idx].zVel = lParList_h.zVel[idx];
	hpLocBuf[idx].xAcc = lParList_h.xAcc[idx];
        hpLocBuf[idx].yAcc = lParList_h.yAcc[idx];
        hpLocBuf[idx].zAcc = lParList_h.zAcc[idx];
        hpLocBuf[idx].mass = lParList_h.mass[idx];
    }

    return cudaSuccess;
}

cudaError_t InitializeParticleLists() {

    //Allocate space for storing entire hpDataset (GPU)
    CHK_ERR(    cudaMalloc(&(gParList_d.xPos), hpDataset->mNumParticles *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(gParList_d.yPos), hpDataset->mNumParticles *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(gParList_d.zPos), hpDataset->mNumParticles *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(gParList_d.xVel), hpDataset->mNumParticles *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(gParList_d.yVel), hpDataset->mNumParticles *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(gParList_d.zVel), hpDataset->mNumParticles *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(gParList_d.xAcc), hpDataset->mNumParticles *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(gParList_d.yAcc), hpDataset->mNumParticles *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(gParList_d.zAcc), hpDataset->mNumParticles *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(gParList_d.mass), hpDataset->mNumParticles *
        sizeof(PRECISION)));

    //Allocate local/working space (GPU)
    CHK_ERR(    cudaMalloc(&(lParList_d.xPos), gpuConfig.localCnt *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(lParList_d.yPos), gpuConfig.localCnt *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(lParList_d.zPos), gpuConfig.localCnt *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(lParList_d.xVel), gpuConfig.localCnt *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(lParList_d.yVel), gpuConfig.localCnt *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(lParList_d.zVel), gpuConfig.localCnt *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(lParList_d.xAcc), gpuConfig.localCnt *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(lParList_d.yAcc), gpuConfig.localCnt *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(lParList_d.zAcc), gpuConfig.localCnt *
        sizeof(PRECISION)));
    CHK_ERR(    cudaMalloc(&(lParList_d.mass), gpuConfig.localCnt *
        sizeof(PRECISION)));

#define CHK_HALLOC(var, size)        \
    do {                             \
        var = new PRECISION [size];  \
        if (var == NULL)             \
            return cudaErrorUnknown; \
    } while (0);

    //Allocate space for storing entire hpDataset (Host)
    CHK_HALLOC(gParList_h.xPos, hpDataset->mNumParticles);
    CHK_HALLOC(gParList_h.yPos, hpDataset->mNumParticles);
    CHK_HALLOC(gParList_h.zPos, hpDataset->mNumParticles);
    CHK_HALLOC(gParList_h.xVel, hpDataset->mNumParticles);
    CHK_HALLOC(gParList_h.yVel, hpDataset->mNumParticles);
    CHK_HALLOC(gParList_h.zVel, hpDataset->mNumParticles);
    CHK_HALLOC(gParList_h.xAcc, hpDataset->mNumParticles);
    CHK_HALLOC(gParList_h.yAcc, hpDataset->mNumParticles);
    CHK_HALLOC(gParList_h.zAcc, hpDataset->mNumParticles);
    CHK_HALLOC(gParList_h.mass, hpDataset->mNumParticles);

    //Allocate local/working space (Host)
    CHK_HALLOC(lParList_h.xPos, hpDataset->mNumParticles);
    CHK_HALLOC(lParList_h.yPos, hpDataset->mNumParticles);
    CHK_HALLOC(lParList_h.zPos, hpDataset->mNumParticles);
    CHK_HALLOC(lParList_h.xVel, hpDataset->mNumParticles);
    CHK_HALLOC(lParList_h.yVel, hpDataset->mNumParticles);
    CHK_HALLOC(lParList_h.zVel, hpDataset->mNumParticles);
    CHK_HALLOC(lParList_h.xAcc, hpDataset->mNumParticles);
    CHK_HALLOC(lParList_h.yAcc, hpDataset->mNumParticles);
    CHK_HALLOC(lParList_h.zAcc, hpDataset->mNumParticles);
    CHK_HALLOC(lParList_h.mass, hpDataset->mNumParticles);

#undef CHK_HALLOC

    ConvertHostToDevice();

    return cudaSuccess;
}

cudaError_t DestroyParticleLists() {
    CHK_ERR(    cudaFree(gParList_d.xPos));
    CHK_ERR(    cudaFree(gParList_d.yPos));
    CHK_ERR(    cudaFree(gParList_d.zPos));
    CHK_ERR(    cudaFree(gParList_d.xVel));
    CHK_ERR(    cudaFree(gParList_d.yVel));
    CHK_ERR(    cudaFree(gParList_d.zVel));
    CHK_ERR(    cudaFree(gParList_d.xAcc));
    CHK_ERR(    cudaFree(gParList_d.yAcc));
    CHK_ERR(    cudaFree(gParList_d.zAcc));
    CHK_ERR(    cudaFree(gParList_d.mass));

    CHK_ERR(    cudaFree(lParList_d.xPos));
    CHK_ERR(    cudaFree(lParList_d.yPos));
    CHK_ERR(    cudaFree(lParList_d.zPos));
    CHK_ERR(    cudaFree(lParList_d.xVel));
    CHK_ERR(    cudaFree(lParList_d.yVel));
    CHK_ERR(    cudaFree(lParList_d.zVel));
    CHK_ERR(    cudaFree(lParList_d.xAcc));
    CHK_ERR(    cudaFree(lParList_d.yAcc));
    CHK_ERR(    cudaFree(lParList_d.zAcc));
    CHK_ERR(    cudaFree(lParList_d.mass));

    delete [] gParList_h.xPos;
    delete [] gParList_h.yPos;
    delete [] gParList_h.zPos;
    delete [] gParList_h.xVel;
    delete [] gParList_h.yVel;
    delete [] gParList_h.zVel;
    delete [] gParList_h.xAcc;
    delete [] gParList_h.yAcc;
    delete [] gParList_h.zAcc;
    delete [] gParList_h.mass;

    delete [] lParList_h.xPos;
    delete [] lParList_h.yPos;
    delete [] lParList_h.zPos;
    delete [] lParList_h.xVel;
    delete [] lParList_h.yVel;
    delete [] lParList_h.zVel;
    delete [] lParList_h.xAcc;
    delete [] lParList_h.yAcc;
    delete [] lParList_h.zAcc;
    delete [] lParList_h.mass;

    return cudaSuccess;
}

cudaError_t CudaInitialize(ParticleDataset * dataset,
    ParticleDataset::Particle * locBuf, NbodyConfig * config,
    GPU_ParticleDataset::GPUConfig gConfig) {

    hpDataset    = dataset;
    hpLocBuf     = locBuf;
    hpConfig     = config;
    gpuConfig    = gConfig;

    int deviceCnt = 0;
    CHK_ERR(    cudaGetDeviceCount(&deviceCnt));
    CHK_ERR(    cudaSetDevice(config->mParams.rank % deviceCnt));

#ifdef _DEBUG
    cout << "[" << config->mParams.rank << "]: cudaSetDevice " <<
        config->mParams.rank % deviceCnt << endl;
#endif

    InitializeParticleLists();

    return cudaSuccess;
}

cudaError_t CudaClean() {

    DestroyParticleLists();

    return cudaSuccess;
}

cudaError_t ComputeParticleAttributes() {

    CHK_ERR(    ConvertHostToDevice());

//    cout << "Launching" << endl;
    ComputeParticleAttributes_Kernel <<<16, 128>>> (
        gParList_d.xPos, gParList_d.yPos, gParList_d.zPos,
        gParList_d.xVel, gParList_d.yVel, gParList_d.zVel,
        gParList_d.xAcc, gParList_d.yAcc, gParList_d.zAcc,
        gParList_d.mass,
        lParList_d.xPos, lParList_d.yPos, lParList_d.zPos,
        lParList_d.xVel, lParList_d.yVel, lParList_d.zVel,
        lParList_d.xAcc, lParList_d.yAcc, lParList_d.zAcc,
        lParList_d.mass,
        hpConfig->mParams.timeRes, hpConfig->mParams.gravConstant,
        gpuConfig.localCnt, gpuConfig.localDisp, hpDataset->mNumParticles);

    CHK_ERR(    cudaDeviceSynchronize());

    CHK_ERR(    ConvertDeviceToHost());

    return cudaSuccess;
}

