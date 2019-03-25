#include "nbody.decl.h"
#include "nbody.h"

char * pConfigFile  = (char *) "nn.conf";
char * pDatasetFile = (char *) NULL;
char * pLibFile     = (char *) "libnn.so";
int    peset;

// Funtion 

#define CHK_ERR(str)                                             \
    do {                                                         \
        cudaError_t ce = str;                                    \
        if (ce != cudaSuccess) {                                 \
            ckout << "Error: " << cudaGetErrorString(ce) << endl; \
        }                                                        \
    } while (0)

char sEMSG[][ECFG_MSG_LEN] = {
    "Success",
    "Unknown argument",
    "Configuration file not specified",
    "Library not specified"
};

void DataLoad(int num, ParticleDataset::Particle *dPar, ParticleDataset::Particle *sPar) {
    memcpy(dPar, sPar, num * sizeof(ParticleDataset::Particle));
}
void ShowConfiguration(NbodyConfig * pConf) {

    ckout << endl << endl << "Cluster env.    \t";
    if (pConf->mParams.cluster) {
        ckout << "yes" << endl;
/*
        len = pConfig->mParams.nodeCnt;
        cout << "\tNum nodes    \t" << len << endl << endl;
        for (idx=0;idx<len;idx++) {
            cout << "\t===== Node# " << idx << "=====" << endl;
            cout << "\tName    \t" 
                 << pConfig->mParams.nodeSettings[idx].nodeName << endl;
            cout << "\tGPU    \t\t";
            if (pConfig->mParams.nodeSettings[idx].useGPU) {
                cout << "yes" << endl;
                cout << "\tNum GPU \t"
                     << pConfig->mParams.nodeSettings[idx].gpuCnt << endl;
            }
            else
                cout << "no" << endl;
            cout << endl;
        }
*/
    }
    else
        ckout << "no" << endl;
    ckout << "time ressolu.\t\t" << pConf->mParams.timeRes << endl;
    ckout << "Duration.\t\t" << pConf->mParams.duration << endl;
//    ckout << "Dataset\t\t\t" << pConf->mParams.initialDatasetFile << endl;
//    ckout << "Library\t\t\t" << pConf->mParams.library << endl;
    ckout << endl << endl;

}

inline int SetPreference(int argc, char ** argv) {
#define IS_OPTION(str) (!strcmp(argv[idx], str))

    int idx = 1;

    while (idx < argc) {
        if IS_OPTION("-c") {
            if ((idx + 1) >= argc)
                return E_FILE_ARG;
            pConfigFile = argv[++idx];
        }
        else if IS_OPTION("--config") {
            if ((idx + 1) >= argc)
                return E_FILE_ARG;
            pConfigFile = argv[++idx];
        }
        else if IS_OPTION("-l") {
            if ((idx + 1) >= argc)
                return E_LIB_ARG;
            pLibFile = argv[++idx];
        }
        else if IS_OPTION("--library") {
            if ((idx + 1) >= argc)
                return E_LIB_ARG;
            pLibFile = argv[++idx];
        }
        else if IS_OPTION("-s") {
            if ((idx + 1) >= argc)
                return E_LIB_ARG;
            peset = atoi(argv[++idx]);
        }

        else {
            return E_UNKNOWN_ARG;
        }

        idx++;
    }

    return E_SUCCESS;

#undef IS_OPTION
}

const char * GetEMSG(int errId) {
    return sEMSG[errId - E_SUCCESS];
}
// Main Proxy

CProxy_Main mainProxy;

class Main : public CBase_Main {
    double startTime;
    int numParticles, penum;
    CProxy_Data mpe, dst;
    ParticleDataset * pDataset;
    NbodyConfig     * pConfig;
    int iter;
    GPU_ParticleDataset::GPUConfig gConfig;
    PRECISION sec, duration, step;
public:
    Main(CkArgMsg *m) {
        int        mErrId;
        bool       isCluster;
        bool       isOshmem = false;
        int        rank;
        int        commSize;
        char       name[218];
        snprintf(name, sizeof(char) * 32, "app_%s",m->argv[0]);
        if ((mErrId = SetPreference(m->argc, m->argv)) != E_SUCCESS) {
            ckout << "Error: Main: "
                 << GetEMSG(mErrId) << endl;

            CkAbort("Error: Main");
        }
// PEs info
//        int penum = CkNumPes();
        penum = peset;
        ckout << "\t\tPENUM: " << peset << endl;  
        iter = 0;
// Read Configuration file
        try {
            pConfig = new NbodyConfig(pConfigFile);
        }
        catch (int errId) {
            ckout << "Error: "
                  << NbodyConfig::GetClassID(errId) << ": "
                  << NbodyConfig::GetEMSG(errId) << endl;

            CkAbort("Error: pConfig allocation");
        }
        PRECISION grav = pConfig->gravCon;

        ckout << "\t\tConfigurtion Complete" << endl;
// Read dataset file
        pDatasetFile = const_cast<char *>(pConfig->GetValue(
            pConfig->GetKeyString(_INITDATASET_)));
        ckout << "DataFile: " << pDatasetFile << endl;
        if (pDatasetFile != NULL) {
            try {
                pDataset     = new ParticleDataset(pDatasetFile);
            }
            catch (int errId) {
                ckout << "Error: "
                     << ParticleDataset::GetClassID(errId) << ": "
                     << ParticleDataset::GetEMSG(errId) << endl;

                CkAbort("Error: pDataset Initialization");
            }
        }
        pDataset->SaveToFile(name);
        ckout << "\t\tDataLoad Complete" << pDataset->mpParticle[2].xPos<< endl;

// Show configuration
        ShowConfiguration(pConfig);
// Initialize parameter

        sec = 0.0;
        numParticles = pDataset->mNumParticles;
        step         = pConfig->mParams.timeRes;
        duration     = pConfig->mParams.duration;
// Spawn PEs
//        dst = CProxy_Data::ckNew(pConfigFile, pDataset, numParticles);
        mpe = CProxy_Data::ckNew(pDataset->mpParticle, numParticles, grav, duration, step, penum, true, 1);
        dst = CProxy_Data::ckNew(pDataset->mpParticle, numParticles, grav, duration, step, penum, false, penum);
        // send out data
        mpe.SendInput(dst);
// Begin the iteration 
        ckout << "Computation Begin" << endl;
        dst.IterBegin(CkCallback(CkReductionTarget(Data, gather_res), mpe)); 

    }
   
    void collect() {
#if 0
        CkReduction::setElement *current = 
            (CkReduction::setElement*) msg->getData();

        while(current !=NULL) {
            ParticleDataset::Particle * result = 
                (ParticleDataset::Particle*) &current->data;
            int localCnt  = result->localCnt;
            int localDisp = result->localDisp;
            int pid       = result->pid;
            memcpy(pDataset->mpParticle+localDisp, result, 
                localCnt * sizeof(ParticleDataset::Particle));
            current = current->next();
        }
#endif
        sec += step;
        iter++;
        if (sec < duration) {
            if (iter % 2) {
                mpe.pauseForLB();
                dst.pauseForLB();
            }
            else {
                mpe.SendInput(dst);
                dst.IterBegin(CkCallback(CkReductionTarget(Data, gather_res), mpe)); 
            }
        }
        else {
            double endTime  = CkWallTimer();
            pDataset->SaveToFile("charmgpu.bin"); 
            CkPrintf("Time: %lf\n", endTime-startTime);
            CkExit();
        }
    }
    void resumeIter(CkReductionMsg *msg) {
        CkPrintf("Resume iteration at step %d\n", iter);
        CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
        bool *result;
        if (current != NULL) {
            result = (bool *) &current->data;
        }
        if (* result == true) 
          mpe.SendInput(dst);
        else
          dst.IterBegin(CkCallback(CkReductionTarget(Data, gather_res), mpe)); 
    }

};

class Data : public CBase_Data {
    NbodyConfig * plConfig;
    ParticleDataset * plDataset;
    ParticleDataset::Particle * localBuf;
    PRECISION sec;
    int localCnt;
    int localDisp;
    int numParticles;
    int penum, peid;
    bool copy, ctype;
    PRECISION grav, duration, step;
    Data_SDAG_CODE
    GPU_ParticleDataset::GPUConfig gConfig;
public:
    Data(ParticleDataset::Particle * pL, int numParticles_, PRECISION grav_, PRECISION duration_, PRECISION step_, int penum_, bool morwpe) : numParticles(numParticles_) {
// Initialize parameter
        usesAtSync = true;
        penum = penum_;
        peid  = thisIndex;
        bool copy = false;
        ctype     = morwpe;
        grav      = grav_;
        duration  = duration_;
        step      = step_;
        sec = 0.0;
// Get data from main Proxy
        plDataset = new ParticleDataset(numParticles); 
        memcpy(plDataset->mpParticle, pL,
            numParticles * sizeof(ParticleDataset::Particle));
//Determine number of particles handled locally.
        int divCnt = numParticles / penum;
        int remCnt = numParticles % penum;
        int total_pe = penum;
        int idx = peid;
//Adjust local particle-cpimnts and displacements.

        if (!remCnt) {
            localCnt = divCnt;
            localDisp = idx * divCnt;
        }
        else {
            if (idx == total_pe-1)
                localCnt = numParticles - (idx * (divCnt + 1));
            else
                localCnt = divCnt + 1;
        localDisp = idx * (divCnt + 1);

        }
        localBuf  = new ParticleDataset::Particle[localCnt];
        localBuf->localCnt  = localCnt;
        localBuf->localDisp = localDisp;
        localBuf->pid       = peid;
        gConfig.localCnt    = localCnt;
        gConfig.localDisp   = localDisp;
        gConfig.pid         = peid;
        gConfig.grav        = grav;
        gConfig.dev         = CkMyPe();
        ckout << "[" << peid << "/"<< penum << "]: " << localDisp << "/" << localCnt << " Data: "<< localBuf[1].xPos<<endl;
//CUDA initialize
//        CHK_ERR(    CudaInitialize(plDataset, localBuf, gConfig));
    }
    void pup(PUP::er &p) {
        CBase_Data::pup(p);
        __sdag_pup(p);
        p|localCnt; p|localDisp; p|penum; p|peid; p|numParticles;
        p|grav; p|duration; p|step;p|sec; p|ctype;

        if (p.isUnpacking()) {
            plDataset = new ParticleDataset(numParticles);
            localBuf  = new ParticleDataset::Particle[localCnt];
        }
        PUParray(p, plDataset->mpParticle, numParticles);
        PUParray(p, localBuf, localCnt);
    }

    Data(CkMigrateMessage*) {}


    void CudaCompute() {
        memcpy(localBuf, (plDataset->mpParticle + localDisp), localCnt * sizeof(ParticleDataset::Particle));
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

        int iIdx, jIdx;
        PRECISION x1Pos, y1Pos, z1Pos;
        PRECISION x1Vel, y1Vel, z1Vel;
        PRECISION x1Acc, y1Acc, z1Acc;
        PRECISION mass1;
        PRECISION x2Pos, y2Pos, z2Pos;
        PRECISION mass2;
        PRECISION force, force_x, force_y, force_z;
        PRECISION radius, radius_s;
        int peid = thisIndex;
        gConfig.step = step;
        ckout << "secs: " << sec << "/" << duration << "\xd";
        //if (sec < duration) {
        CHK_ERR(    ComputeParticleAttributes(plDataset, localBuf, gConfig, step, grav, CkMyPe(), peid, localCnt, localDisp, numParticles));
        //    sec += step;
       // } else {
       //     mainProxy.done(thisIndex);
       // }
 

    }
    void gather_res(CkReductionMsg *msg) {

        CkReduction::setElement *current =
            (CkReduction::setElement*) msg->getData();

        while(current !=NULL) {
            ParticleDataset::Particle * result =
                (ParticleDataset::Particle*) &current->data;
            int localCnt  = result->localCnt;
            int localDisp = result->localDisp;
            int pid       = result->pid;
            memcpy(plDataset->mpParticle+localDisp, result,
                localCnt * sizeof(ParticleDataset::Particle));
            current = current->next();
        }
        mainProxy.collect();
    }
    void pauseForLB() {
        CkPrintf("Element %d pause for LB on PE %d\n", thisIndex, CkMyPe());
        AtSync();
    }
    void ResumeFromSync() {
        CkCallback cbld(CkReductionTarget(Main, resumeIter), mainProxy);
        contribute(sizeof(bool), &ctype, CkReduction::set, cbld);
       // CkCallback cbld(CkReductionTarget(Main, collect), mainProxy);
        //contribute(localCnt*sizeof(ParticleDataset::Particle), localBuf, CkReduction::set, cbld);
        //IterBegin(plDataset->mpParticle, numParticles,
        //    CkCallback(CkReductionTarget(Main, collect), mainProxy));
    }
};


#include "nbody.def.h"
