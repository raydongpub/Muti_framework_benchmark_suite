#include "nbody_main.h"
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
    if (plConfig->mParams.rank == MPI_NN_ROOT) {
        numParticles = plDataset->mNumParticles;
    }

    MPI_Bcast(&numParticles, 1, MPI_INT, MPI_NN_ROOT, MPI_COMM_WORLD);

    if (plConfig->mParams.rank != MPI_NN_ROOT)
        plDataset = new ParticleDataset(numParticles,oshm);

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

    cout << "[" << plConfig->mParams.rank << "]: "
        << localDisp << "/" << localCnt << endl;
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


    CollectiveInitialize();

    cout << "[" << plConfig->mParams.rank << "/"
       << plConfig->mParams.commSize << "]: "
       << localCnt << endl;

    PRECISION step         = plConfig->mParams.timeRes;
    PRECISION duration     = plConfig->mParams.duration;
    PRECISION grav         = plConfig->mParams.gravConstant;

    for (sec=0.0;sec<duration;sec+=step) {

        PRECISION x1Pos, y1Pos, z1Pos;
        PRECISION x1Vel, y1Vel, z1Vel;
        PRECISION x1Acc, y1Acc, z1Acc;
        PRECISION mass1;
        PRECISION x2Pos, y2Pos, z2Pos;
        PRECISION mass2;
        PRECISION force, force_x, force_y, force_z;
        PRECISION radius, radius_s;

        memcpy(localBuf, plDataset->mpParticle + localDisp,
            localCnt * sizeof(ParticleDataset::Particle));

        for (int iIdx=0;iIdx<localCnt;iIdx++) {
            SET_PARTICLE(1, iIdx);

            force_x = 0; force_y = 0; force_z = 0;
            for (int jIdx=0;jIdx<plDataset->mNumParticles;jIdx++) {
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

        if (!plConfig->mParams.rank)
            cout << "secs: " << sec << "/" << duration << "\xd";

        MPI_Allgatherv(localBuf, localCnt, mpiParticleType,
            plDataset->mpParticle, cntList, dispList, mpiParticleType,
            MPI_COMM_WORLD);
    }

    CollectiveClean();
}

int LibCleanUp(void) {

    if (!plConfig->mParams.rank)
        plDataset->SaveToFile("mpi.bin");

    return 0;
}

