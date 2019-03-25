#include "nbody_main.h"
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

NbodyConfig               * plConfig;
ParticleDataset           * plDataset;
ParticleDataset           * shDataset;
PRECISION                   sec = 0.0;
int                       * numParticles;

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
        shmem_int_get(numParticles, numParticles, 1, 0);
    }
    shmem_barrier_all();

    shDataset = new ParticleDataset(*numParticles, osh);

    if (!mid) {
        memcpy(shDataset->mpParticle, plDataset->mpParticle,
            (*numParticles * sizeof(ParticleDataset::Particle)));
    }
    shmem_barrier_all();
    if (mid) {
        shmem_getmem(shDataset->mpParticle, shDataset->mpParticle,           (*numParticles * sizeof(ParticleDataset::Particle)), 0);
    }    
    shmem_barrier_all();

    //Determine number of particles handled locally.
    int total_pe = _num_pes();
    int idx = _my_pe();
    int divCnt = *numParticles / total_pe;
    int remCnt = *numParticles % total_pe;
    //Adjust local particle-cpimnts and displacements.

    if (!remCnt) {
        *localC = divCnt;
        *localD = idx * divCnt;
    }
    else {
        if (idx == total_pe-1)
            *localC = *numParticles - (idx * (divCnt + 1));
        else
            *localC = divCnt + 1;
        *localD = idx * (divCnt + 1);
       
    }

    localB  = new ParticleDataset::Particle[*localC];

    printf("[%d]: %d/%d\n", plConfig->mParams.rank, 
        *localD, *localC);

    return localB;
}

inline void CollectiveClean(ParticleDataset::Particle *localB) {

    delete [] localB;
}

void LibEntry(int argc, char **argv) {
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

    ParticleDataset::Particle * localBuf;
    int                         localCnt;
    int                         localDisp;
    int                         idx;

    localBuf = (ParticleDataset::Particle *)CollectiveInitialize(
        &localCnt, &localDisp);

    cout << "[" << plConfig->mParams.rank << "/"
       << plConfig->mParams.commSize << "]: "
       << localCnt << " total: " << *numParticles << endl;

    PRECISION step         = plConfig->mParams.timeRes;
    PRECISION duration     = plConfig->mParams.duration;
    PRECISION grav         = plConfig->mParams.gravConstant;
    idx = _my_pe();
    printf("[%d]: %d/%d\n", plConfig->mParams.rank, 
        localDisp, localCnt);

//Info
    //cout << "Length of structure is: " << sizeof(ParticleDataset::Particle) << endl;

    for (sec=0.0;sec<duration;sec+=step) {

        PRECISION x1Pos, y1Pos, z1Pos;
        PRECISION x1Vel, y1Vel, z1Vel;
        PRECISION x1Acc, y1Acc, z1Acc;
        PRECISION mass1;
        PRECISION x2Pos, y2Pos, z2Pos;
        PRECISION mass2;
        PRECISION force, force_x, force_y, force_z;
        PRECISION radius, radius_s;

        memcpy(localBuf, (shDataset->mpParticle + localDisp), 
            localCnt * sizeof(ParticleDataset::Particle));

        for (int iIdx=0;iIdx<localCnt;iIdx++) {
            SET_PARTICLE(1, iIdx);

            force_x = 0; force_y = 0; force_z = 0;
            for (int jIdx=0;jIdx<shDataset->mNumParticles;jIdx++) {
                if (jIdx != (localDisp + iIdx)) {
                    SET_IPARTICLE(2, jIdx);

                    radius_s  = ((x2Pos - x1Pos) * (x2Pos - x1Pos)) +
                                ((y2Pos - y1Pos) * (y2Pos - y1Pos)) +
                                ((z2Pos - z1Pos) * (z2Pos - z1Pos));
                    radius    = sqrt(radius_s);
                    force     = (grav * mass1 * mass2) / radius_s;
                    force_x  += force * ((x2Pos - x1Pos) / radius);
                    force_y  += force * ((y2Pos - y1Pos) / radius);
                    force_z  += force * ((z2Pos - z1Pos) / radius);
                }
            }

            x1Acc    = force_x / mass1;
            y1Acc    = force_y / mass1;
            z1Acc    = force_z / mass1;
            x1Vel    += x1Acc * step;
            y1Vel    += y1Acc * step;
            z1Vel    += z1Acc * step;
            x1Pos    += x1Vel * step;
            y1Pos    += y1Vel * step;
            z1Pos    += z1Vel * step;

            localBuf[iIdx].xPos = x1Pos;
            localBuf[iIdx].yPos = y1Pos;
            localBuf[iIdx].zPos = z1Pos;
            localBuf[iIdx].xVel = x1Vel;
            localBuf[iIdx].yVel = y1Vel;
            localBuf[iIdx].zVel = z1Vel;
            localBuf[iIdx].xAcc = x1Acc;
            localBuf[iIdx].yAcc = y1Acc;
            localBuf[iIdx].zAcc = z1Acc;
        }

        if (!idx)
            cout << "secs: " << sec << "/" << duration << "\xd";

        shmem_putmem((shDataset->mpParticle + localDisp), 
            localBuf, (localCnt  * 
            sizeof(ParticleDataset::Particle)), 0);
        shmem_barrier_all();
        if (idx) {
/*            shmem_getmem(shDataset->mpParticle, 
                shDataset->mpParticle, (*numParticles) * 
                sizeof(ParticleDataset::Particle), 0);*/
            shmem_getmem(shDataset->mpParticle, 
                shDataset->mpParticle, ((*numParticles) 
                * sizeof(ParticleDataset::Particle)), 0);
        }
        shmem_barrier_all();
    }

    CollectiveClean(localBuf);
}

int LibCleanUp(void) {

    if ((_my_pe()) == 0)
        shDataset->SaveToFile("oshm.bin");
    shfree(numParticles);
    return 0;
}

