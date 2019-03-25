#include "nbody_main.h"
#include "ParticleDataset.h"
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

NbodyConfig               * plConfig;
ParticleDataset           * plDataset;
ParticleDataset           * shDataset;
PRECISION                   sec = 0.0;
int                       * numParticles;

int LibSetup(NbodyConfig * config,
        ParticleDataset * dataset) {
    pe_id my_id;
    checkError(    ivmGetMyId(&my_id), "ivmGetmyId()");
    plConfig  = config;
    plDataset = dataset;
   return 0;
}

inline void * CollectiveInitialize(int * localC, int * localD, ivm_params * ivmp) {
    ParticleDataset::Particle * localB;
// IVM Definition
    int mid = ivmp->work_id;

    //Determine number of particles handled locally.
    int total_pe = ivmp->num_pes;
    int idx = mid - 1;
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

void LibEntry_IVM(ivm_params * ivmparams) {
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

    bool * sync_var;
    int  * iter_fac;
    int    sizeParticles;

// IVM Parameter
    ivm_params * ivmp = ivmparams;
    int peid = ivmp->my_id;
    int num_pes = ivmp->num_pes;
    int sync_size = num_pes * sizeof(bool);

    if (peid == 0) {
        bool sync_flag = false;
// Synchronize var
        checkError(    ivmMalloc((void **) &sync_var, sync_size, 
            "SYNCNB"), "ivmMalloc(SYNCNB)");
        for (int i=0; i<num_pes; i++)
            sync_var[i] = false;
        checkError(    ivmMalloc((void **) &iter_fac, sizeof(int),             "ITERNB"), "ivmMalloc(ITERNB)");
        *iter_fac = 0;

        checkError(    ivmMalloc((void **) &numParticles, 
             sizeof(int), "NUM"), "ivmMalloc(NUM)");
// Sync-up the number of total particles of main process
        *numParticles = plDataset->mNumParticles;
        sizeParticles = *numParticles * 
            sizeof(ParticleDataset::Particle);
// Construct a public shared Dataset object
        shDataset = new ParticleDataset(*numParticles);
// Allocate Memory for Particle variable in Dataset objec
        checkError(    ivmMalloc((void **) 
            &(shDataset->mpParticle),sizeParticles, "DATASET"), 
            "ivmMalloc(DATASET)");
        memcpy(shDataset->mpParticle, plDataset->mpParticle,
            sizeParticles); 

// Create node and pe
        checkError(    ivmCreateNode(IVM_THIS_NODE, 
            IVM_THIS_SERVICE, ivm_rdma, ivm_cpu, &(ivmp->node[0])),            "ivmCreateNode(own)");

/*        checkError(    ivmCreateNode("nps3", IVM_THIS_SERVICE, 
            ivm_rdma, &(ivmp->node[1])), "ivmCreateNode(nps103)");*/ 


        PRECISION step         = plConfig->mParams.timeRes;
        PRECISION duration     = plConfig->mParams.duration;
        PRECISION grav         = plConfig->mParams.gravConstant;

        for (sec=0.0;sec<duration;sec+=step) {
            checkError(    ivmSetMaxProcessID(num_pes * (*iter_fac+1)),                "ivmSetMaxProcessID()");
            checkError(    ivmCreateProcess(ivmp->node[0], 
                IVM_THIS_BINARY, 2, ivmp->argc, ivmp->argv, 
                ivmp->pe), "ivmCreateProcess(own)");
            cout << "secs: " << sec << "/" << duration << "\xd";
            do {
                sync_flag = false;
                for (int i=0; i<num_pes; i++) {
                    if (sync_var[i] == false) {
                        sync_flag = true;
                    }
                }   
            } while(sync_flag); 
            for (int i=0; i<num_pes; i++)
                sync_var[i] = false;
            *iter_fac = (*iter_fac + 1);     
        }
// IVM-Wait
    }
    else {
// Sync the variable and data for all PEs
        checkError(    ivmMap((void **) &sync_var, sync_size, 
            "SYNCNB"),"ivmMap(SYNCNB)");
        checkError(    ivmMap((void **) &iter_fac, sizeof(int), 
            "ITERNB"), "ivmMap(ITERNB)");
        checkError(    ivmMap((void **) &numParticles, 
            sizeof(int), "NUM"), "ivmMap(NUM)");
        sizeParticles = *numParticles * 
            sizeof(ParticleDataset::Particle);
// Construct a public shared Dataset object
        shDataset = new ParticleDataset(*numParticles);
// Map to the Dataset object
        checkError(    ivmMap((void **) &(shDataset->mpParticle), 
            sizeParticles, "DATASET"), "ivmMap(DATASET)");

        if (peid > num_pes) {
            checkError(    ivmSyncGet("ITERNB", iter_fac, 
                sizeof(int)), "ivmSyncGet(ITERNB)");   
// Sync Get all new data from main PE
            int syntotal = *numParticles * 
                sizeof(ParticleDataset::Particle);
            checkError(    ivmSyncGet("DATASET", 
                (shDataset->mpParticle), syntotal), 
                "ivmSyncGet(DATASET)");

        }

// Create node and pe
        checkError(    ivmCreateNode(IVM_THIS_NODE, 
            IVM_THIS_SERVICE, ivm_rdma, ivm_cpu, &(ivmp->node[0])),            "ivmCreateNode(own)");
// Get work ID
        ivmp->iter    = *iter_fac;
        ivmp->work_id = peid - *iter_fac * num_pes;
        int workid    = ivmp->work_id;        
   
        localBuf = (ParticleDataset::Particle *)
            CollectiveInitialize(&localCnt, &localDisp, ivmp);
    
        cout << "[" << plConfig->mParams.rank << "/"
            << plConfig->mParams.commSize << "]: "
            << localCnt << " total: " << *numParticles << endl;

        PRECISION step         = plConfig->mParams.timeRes;
        PRECISION duration     = plConfig->mParams.duration;
        PRECISION grav         = plConfig->mParams.gravConstant;

// Get my pe_id
        idx = ivmp->my_id;
        printf("[%d]: %d/%d\n", plConfig->mParams.rank, 
            localDisp, localCnt);

//Info
    //cout << "Length of structure is: " << sizeof(ParticleDataset::Particle) << endl;


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
            for (int jIdx=0;jIdx<shDataset->mNumParticles;
                jIdx++) {
                if (jIdx != (localDisp + iIdx)) {
                    SET_IPARTICLE(2, jIdx);

                    radius_s  = ((x2Pos - x1Pos) * (x2Pos - x1Pos))                        + ((y2Pos - y1Pos) * (y2Pos - y1Pos)) + 
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
// Sync back the data to main PE
        int synsize=localCnt*sizeof(ParticleDataset::Particle);
        checkError(    ivmSyncPut("DATASET", 
            (shDataset->mpParticle + localDisp),synsize), 
            "ivmSyncPut(DATASET)");
        sync_var[workid-1] = true;
        checkError(    ivmSyncPut("SYNCNB", &(sync_var[workid-1])                , sizeof(bool)), "ivmSyncPut(SYNCNB)");
        if (workid < num_pes) {
            checkError(    ivmCreateProcess(ivmp->node[0], 
                IVM_THIS_BINARY, 1, ivmp->argc, ivmp->argv, 
                ivmp->pe), "ivmCreateProcess(own)");
        }
        CollectiveClean(localBuf);
    }
}

int LibCleanUp(void) {
    pe_id my_id;
    checkError(    ivmGetMyId(&my_id), "ivmGetMyId()");
    if ((my_id) == 0)
        shDataset->SaveToFile("ivmpc.bin");
//    shfree(numParticles);
    return 0;
}

