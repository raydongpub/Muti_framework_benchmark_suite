#include "nbody_main.h"
#include <dlfcn.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sys/time.h>
#ifdef _OSHM_MOD
#include <shmem.h>
#endif //_OSHM_MOD

#define E_SUCCESS     0xFF00
#define E_UNKNOWN_ARG 0XFF01
#define E_FILE_ARG    0xFF02
#define E_LIB_ARG     0xFF03


//For building the version that uses various static libraries.
#ifdef _USE_STATIC_LIB
#ifdef _STATIC_MOD_ST
#elif  _STATIC_MOD_MPI
#elif  _STATIC_MOD_MPICUDA
#endif //_STATIC_MOD_XXXXX
#endif //_USE_STATIC_LIB

using namespace std;

char            * pConfigFile  = (char *) "nn.conf";
char            * pDatasetFile = (char *) NULL;
char            * pLibFile     = (char *) "libnn.so";
int               blocks, threads;

NbodyConfig     * pConfig  = NULL;
ParticleDataset * pDataset = NULL;

char sEMSG[][ECFG_MSG_LEN] = {
    "Success",
    "Unknown argument",
    "Configuration file not specified",
    "Library not specified"
};

inline void ShowConfiguration() {

    cout << endl << endl << "Cluster env.    \t";
    if (pConfig->mParams.cluster) { 
        cout << "yes" << endl;
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
        cout << "no" << endl;
    cout << "time ressolu.\t\t" << pConfig->mParams.timeRes << endl;
    cout << "Duration.\t\t" << pConfig->mParams.duration << endl;
    cout << "Dataset\t\t\t" << pConfig->mParams.initialDatasetFile << endl;
    cout << "Library\t\t\t" << pConfig->mParams.library << endl;
    cout << endl << endl;
    
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
        else if (IS_OPTION("-b")) {
            blocks = atoi(argv[++idx]);
        }
        else if (IS_OPTION("-t")) {
            threads = atoi(argv[++idx]);
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

int main(int argc, char ** argv) {

    int        mErrId;
    timeval    time_begin, time_end;
    double     time_period;
    //For MPI only
    bool       isCluster;
    bool       isOshmem;
    int        rank;
    int        commSize;
    char       name[218];

    snprintf(name, sizeof(char) * 32, "app_%s",argv[0]);
    if ((mErrId = SetPreference(argc, argv)) != E_SUCCESS) {
        cerr << "Error: Main: "
             << GetEMSG(mErrId) << endl;
        exit(1);
    }

    /**
     * Read configuration file.
     */
    try {
        pConfig = new NbodyConfig(pConfigFile);
    }
    catch (int errId) {
        cerr << "Error: "
             << NbodyConfig::GetClassID(errId) << ": " 
             << NbodyConfig::GetEMSG(errId) << endl;
        exit(1);
    }

    /**
     * Determine whether this is OpenShmem.
     */
    char * isOshmemStr = const_cast<char *>(pConfig->GetValue(pConfig->GetKeyString(_OSHMEM_)));
    if (!strcmp(isOshmemStr, NBC_TRUE)) {
        isOshmem = true;
        cout << endl << endl << "This is OpenShmem!"<< endl << endl;
    }
    else {
        isOshmem = false;
        cout << endl << endl << "This is not OpenShmem!" << *isOshmemStr << endl << endl;
    }

    /**
     * Determine whether this is a distributed computation.
     */
    char * isClusterStr = const_cast<char *>(pConfig->GetValue(
        pConfig->GetKeyString(_CLUSTER_)));

    if (!strcmp(isClusterStr, NBC_TRUE) && !isOshmem) {
        cout << "***** MPI_MOD *****" << endl;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &commSize);

        isCluster = true;
        pConfig->mParams.rank     = rank;
        pConfig->mParams.commSize = commSize;
    }
#ifdef _OSHM_MOD
    else if (!strcmp(isClusterStr, NBC_TRUE) && isOshmem) {
        cout << "***** OSHM_MOD *****" << endl;
        isCluster = true;
        start_pes(0);
        rank = _my_pe();
        commSize = _num_pes();
        pConfig->mParams.rank     = rank;
        pConfig->mParams.commSize = commSize;
    }
#endif //_OSHM_MOD
    else if (!strcmp(isClusterStr, NBC_FALSE)) {
        rank     = 0;
        commSize = 1;

        isCluster = false;
        pConfig->mParams.rank     = 0;
        pConfig->mParams.commSize = 1;
    }
    else {
        rank     = 0;
        commSize = 1;

        isCluster = false;
        pConfig->mParams.rank     = 0;
        pConfig->mParams.commSize = 1;
    }
    
    /**
     * Read dataset file.
   */ 
    if ((!isCluster) || (!rank)) {
        /**
         * Note: In MPI job, we will allocate space for dataset only
         *       in the root. We leave allocations to the library
         *       on the other ranks. We do not want to bind to a
         *       specific implementation. Library must be free
         *       to allocate working space as it wish.
         */
        printf("kernel Config, blocks: %d, threads: %d\n", blocks, threads);
        pDatasetFile = const_cast<char *>(pConfig->GetValue(
            pConfig->GetKeyString(_INITDATASET_)));

        if (pDatasetFile != NULL) {
            try {
                pDataset     = new ParticleDataset(pDatasetFile);
            }
            catch (int errId) {
                cerr << "Error: "
                     << ParticleDataset::GetClassID(errId) << ": "
                     << ParticleDataset::GetEMSG(errId) << endl;
                exit(1);
            }
        }
//        pDataset->SaveToFile(name);
    }

    /**
     * Retrieve and execute the specified computation library.
     */
#ifndef _USE_STATIC_LIB
    void        * dlHandle;
    LibSetupFn    libSetup;
    LibEntryFn    libEntry;
    LibCleanUpFn  libCleanUp;

    pLibFile = const_cast<char *>(pConfig->GetValue(
                                  pConfig->GetKeyString(_LIBRARY_)));
    if (pLibFile != NULL) {
        if (!rank)
            ShowConfiguration();

        dlHandle   = dlopen(pLibFile, RTLD_NOW);
        if (dlHandle != NULL) {
            libEntry   = (LibEntryFn) dlsym(dlHandle, "LibEntry");
            libCleanUp = (LibCleanUpFn) dlsym(dlHandle, "LibCleanUp");

            libSetup(pConfig, pDataset);

            if (isCluster && !isOshmem)
                MPI_Barrier(MPI_COMM_WORLD);
#ifdef _OSHM_MOD
            else if (isOshmem)
                shmem_barrier_all();
#endif //_OSHM_MOD
            libEntry(NULL);
            libCleanUp();
            dlclose(dlHandle);
        }
        else {
            cerr << "Error: Main: " << dlerror() << endl;
        }

    }
#else
    if (!rank)
        ShowConfiguration();
//Info
    cout << endl << "Using Static!" << endl << endl;
    LibSetup(pConfig, pDataset);

    if (isCluster && !isOshmem)
        MPI_Barrier(MPI_COMM_WORLD);
#ifdef _OSHM_MOD
    else if (isOshmem)
        shmem_barrier_all();
#endif //_OSHM_MOD
    gettimeofday(&time_begin,NULL);
        LibEntry(argc, argv, blocks, threads);
    gettimeofday(&time_end,NULL);
//Info
    cout << endl << "Get through!" <<endl;   
    LibCleanUp();
#endif
    cout << "Clean Up Successfully!" << endl << endl;
    time_period = (time_end.tv_sec + time_end.tv_usec * 1e-6) - (time_begin.tv_sec + time_begin.tv_usec * 1e-6);
    printf("Time: %lf\n", time_period);
    /**
     * Finalize, if this is a distributed computation.
     */
    if (isCluster && !isOshmem) {
//info 
        cout << "Is Cluster!" << endl;
        MPI_Finalize();
    }

    if ((!isCluster) || (!rank)) {
        delete pDataset;
    }
    delete pConfig;
}

