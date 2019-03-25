#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <iostream>
#include "cuda_runtime_api.h"
#include "nnMpiCudaKernel.cuh"
#include "ParticleDataset.h"

using namespace std;

/**
 * @Operation mode: the program can either operate in master or worker mode.
 *                  The master is created through the job submission process
 *                  by the user whereas the workers are in-turn created by
 *                  the master. There will always be only one master per
 *                  job submission.
 */

#define OP_STR_WORKER  "worker"
#define OP_MODE_MASTER 0x0a
#define OP_MODE_WORKER 0x0b
int op_mode = OP_MODE_MASTER;


/**
 * @Error message
 */

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

/**
 * @Common MPI attributes
 */
#define MPI_NN_ROOT 0
#ifndef _DOUBLE_PRECISION
#define MPI_PRECISION MPI_FLOAT
#else //_DOUBLE_PRECISION
#define MPI_PRECISION MPI_DOUBLE
#endif //_DOUBLE_PRECISION

int parent_rank;
int rank;
int comm_size;
bool verify = false;


/**
 * @Process attributes
 */
int max_pe;
int pe_node;
int pe_dim;
int pe_per_node;
int wid;
int mat_dim;
int mat_size;

/**
 * @Data type
 */
#ifndef _DOUBLE_PRECISION
#define PRECISION float
#else
#define PRECISION double
#endif

#define NUM_WORKER_PROCS 1

/**
 * @Process attributes
 */
PRECISION * vec_a,   * vec_b,   * vec_c,   * vec_c_cpu;
PRECISION * vec_a_d, * vec_b_d, * vec_c_d;


// MPI data type definition
MPI_Datatype      mpiParticleType;
int               structCount;
int             * structBlock;
MPI_Aint        * structDisplacement;
MPI_Datatype    * structDatatype;

/**
 * @Datatype definition in MPI
 */


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



/**
 * @nbody config
 */
char            * pConfigFile  = (char *) "nn.conf";
char            * pDatasetFile = (char *) NULL;
char            * pLibFile     = (char *) "libnn.so";

NbodyConfig     * pConfig  = NULL;
ParticleDataset * pDataset = NULL;
ParticleDataset * plDataset = NULL;
//CUDA
GPU_ParticleDataset::GPUConfig gConfig;


/**
 * @message functions
 */
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
    }
    else
        cout << "no" << endl;
    cout << "Time ressolu.\t\t" << pConfig->mParams.timeRes << endl;
    cout << "Duration.\t\t" << pConfig->mParams.duration << endl;
    cout << "Dataset\t\t\t" << pConfig->mParams.initialDatasetFile << endl;
    cout << "Library\t\t\t" << pConfig->mParams.library << endl;
    cout << "PE_NODE\t\t\t" << pe_node << endl;
    cout << endl << endl;
}

const char * GetEMSG(int errId) {
    return sEMSG[errId - E_SUCCESS];
}

/**
 * @Error check functions list
 */


inline void checkError(int ret, const char * str) {
    if (ret != 0) {
        cerr << "Error: " << str << endl;
        exit(-1);
    }
}

inline void CheckCUDAError (cudaError_t ce) {
    if (ce != cudaSuccess) {
        cout << "CUDA_ERROR: " << cudaGetErrorString(ce) << endl;
        exit(0);
    }
}

inline void * CollectiveInitialize(int * localC, int * localD, int numPar, int workid, int total_pe, bool flag) {
    ParticleDataset::Particle * localB;
// IVM Definition
    int idx = workid;

    //Determine number of particles handled locally.
    int divCnt = numPar / total_pe;
    int remCnt = numPar % total_pe;
    //Adjust local particle-cpimnts and displacements.

    if (!remCnt) {
        *localC = divCnt;
        *localD = idx * divCnt;
    }
    else {
        if (idx == total_pe-1)
            *localC = numPar - (idx * (divCnt + 1));
        else
            *localC = divCnt + 1;
        *localD = idx * (divCnt + 1);

    }
    if (flag)
        localB  = new ParticleDataset::Particle[*localC];

    return localB;
}

inline void CollectiveClean(ParticleDataset::Particle *localB) {

    delete [] localB;
}

/**
 * @Routines for the master and workers: See the implementations below.
 *
 */
int MasterRoutine(int argc, char ** argv);
int WorkerRoutine(int argc, char ** argv);

int SetOperationMode(int argc, char ** argv) {

    bool matched   = false;
    for (int idx=0;idx<argc;idx++) {
        if (!strcmp(argv[idx], OP_STR_WORKER)) {
            op_mode     = OP_MODE_WORKER;
            matched     = true;
            parent_rank = atoi(argv[idx + 1]);
            wid         = atoi(argv[idx + 2]);
            pConfigFile = argv[idx + 3];
        } 
        if (!strcmp(argv[idx], "-pe") && op_mode == OP_MODE_MASTER) {
            max_pe  = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-pn") && op_mode == OP_MODE_MASTER) {
            pe_node = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-pp") && op_mode == OP_MODE_MASTER) {
            pe_per_node = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-v") && op_mode == OP_MODE_MASTER) {
           verify   = true;
        }
        if (!strcmp(argv[idx], "-c") && op_mode == OP_MODE_MASTER) {
           pConfigFile = argv[idx+1];
           cout << "\t\t Init: " << wid << " by: " << parent_rank << endl;
        }
        if (!strcmp(argv[idx], "-l") && op_mode == OP_MODE_MASTER) {
           pLibFile = argv[idx+1];
        }

    }
    if (!matched)
        op_mode = OP_MODE_MASTER;
}

int main(int argc, char ** argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    SetOperationMode(argc, argv);

    // datatype definition
    DefineDatatype();

    // Read configuration file
    try {
        pConfig = new NbodyConfig(pConfigFile);
    }
    catch (int errId) {
        cerr << "Error: "
             << NbodyConfig::GetClassID(errId) << ": "
             << NbodyConfig::GetEMSG(errId) << endl;
        exit(1);
    }

    int ret;
    switch(op_mode) {
    case OP_MODE_MASTER:
        ret = MasterRoutine(argc, argv);
        break;
    case OP_MODE_WORKER:
        ret = WorkerRoutine(argc, argv);
        break;
    }


    return 0;
}

/**
 * @Routine for the master:
 */

int MasterRoutine(int argc, char ** argv) {

    MPI_Barrier(MPI_COMM_WORLD);

    char rank_str[8], wid_str[8];
    snprintf(rank_str, 8, "%d", rank);

    // Control PE master PE rank = 0
    int count = 0;  
    int count_buf[3];   
    int child_buf[2];
    int control_buf[4];
    int compute_info[2];
    bool end = false;
    int rev_id = 0;
    timeval t_b, t_e;
    double  t_p;
    MPI_Request request; 
    MPI_Status  status;
#if 0
    // one-sided communication buffer create
    MPI_Win win;
    MPI_Win_create(control_buf, pe_node, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
#endif
    memset(count_buf, 0, 3);
    memset(child_buf, 0, 2);
    memset(control_buf, 0, 4);
    memset(compute_info, 0, 2);

    if (rank == 0 ) {
        // Read dataset file
        /**
         * Note: In MPI job, we will allocate space for dataset only
         *       in the root. We leave allocations to the library
         *       on the other ranks. We do not want to bind to a
         *       specific implementation. Library must be free
         *       to allocate working space as it wish.
         */
        pDatasetFile = const_cast<char *>(pConfig->GetValue(
            pConfig->GetKeyString(_INITDATASET_)));
        gettimeofday(&t_b, NULL);
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
        ShowConfiguration();
// Control PE (MPE)
        // control parameter 
        bool iter_switch = true;
        int numParticles = pDataset->mNumParticles;     
        plDataset = new ParticleDataset(pDatasetFile);
        MPI_Bcast(&numParticles, 1, MPI_INT, MPI_NN_ROOT, MPI_COMM_WORLD);

        PRECISION step         = pConfig->mParams.timeRes;
        PRECISION duration     = pConfig->mParams.duration;
        PRECISION grav         = pConfig->mParams.gravConstant;
        PRECISION sec;
        for (sec=0.0;sec<duration;sec+=step) {
            iter_switch = true;
            memcpy(pDataset->mpParticle, plDataset->mpParticle, numParticles * sizeof(ParticleDataset::Particle));
            count = 0;
            memset(count_buf, 0, 3);
            memset(child_buf, 0, 2);
            memset(control_buf, 0, 4);
            memset(compute_info, 0, 2);
            while (count < max_pe) {
                MPI_Recv(control_buf, 4, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
#if 0
            MPI_Win_fence(0, win);
                cout << "==============" << endl << "Show Status: " << local_buf[i] << endl;
                cout << "==============" << endl << endl;
#endif
                rev_id = status.MPI_SOURCE;
                if (rev_id == -1) {
                    cout << "Error: Master PE Monitor Receiving Error!" << endl;
                    exit(0);
                } 
                // control_buf[0] indicates if task is completed by master PE, 2 yes, 1 no
                if (control_buf[0] == 1) {
                    // count_buf[1] indicates control PE accept a new request of child PE 
                    count_buf[1] = 1;
                    // count_buf[2] indicates the WID for new child PE
                    count_buf[2] = count; 
                    // send WID
                    MPI_Send(count_buf, 3, MPI_INT, rev_id, 0, MPI_COMM_WORLD);
                    // receive result data
                    if (control_buf[2] == 1) {
                        /**********************************************************************
                         * Identify work portion
                         **********************************************************************/
                        for (int k=0; k<pe_per_node; k++) {
                            int tmpid = control_buf[1] + k;
                            int localC, localD;
                            void * tptr = CollectiveInitialize(&localC, &localD, numParticles, tmpid, max_pe, false);
                            if (tmpid < max_pe) {
                                //Receive the resutl dataset
                                MPI_Recv(plDataset->mpParticle+localD, localC, mpiParticleType, rev_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            }
                        }


                    } 
                    /**********************************************************************
                     * Identify work portion
                     **********************************************************************/
                    wid = count;
                    if (control_buf[3] == 1) {
                        MPI_Send(pDataset->mpParticle, pDataset->mNumParticles,
                            mpiParticleType, rev_id, 0, MPI_COMM_WORLD); 
                        control_buf[3] = 0;
                    }
                    count += pe_per_node;
                    control_buf[0] = 0;
                    control_buf[1] = 0;
                    control_buf[2] = 0;
                       
                }

            }        
            // receive the final dataset before ending
            for (int i=1; i< pe_node; i++) {
                // receive control_buf to have final data WID
                MPI_Recv(control_buf, 4, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                count_buf[1] = 2;
                // send end flag to WPEs
                MPI_Send(count_buf, 3, MPI_INT, i, 0, MPI_COMM_WORLD);
                    for (int k=0; k<pe_per_node; k++) {
                        int tmpid = control_buf[1] + k;
                        int localC, localD;
                        void * tptr = CollectiveInitialize(&localC, &localD, numParticles, tmpid, max_pe, false);
                        if (tmpid < max_pe) {
                            //Receive result data from WPEs
                            MPI_Recv(plDataset->mpParticle+localD, localC, mpiParticleType, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                    }
            }
            //cout << "secs: " << sec << "/" << duration << "\xd";
            cout << "secs: " << sec << "/" << duration << endl;

        }
        plDataset->SaveToFile("mpi_ld_cuda.bin");

    } else {
        // dataset info
        int numParticles = 0;
        MPI_Bcast(&numParticles, 1, MPI_INT, MPI_NN_ROOT, MPI_COMM_WORLD);
        pDataset  = new ParticleDataset(numParticles);
        plDataset = new ParticleDataset(numParticles);
        PRECISION step         = pConfig->mParams.timeRes;
        PRECISION duration     = pConfig->mParams.duration;
        PRECISION grav         = pConfig->mParams.gravConstant;
        PRECISION sec;
        bool iter_switch = true;
        for (sec=0.0;sec<duration;sec+=step) {
            iter_switch = true;
            memset(count_buf, 0, 3);
            memset(child_buf, 0, 2);
            memset(control_buf, 0, 4);
            memset(compute_info, 0, 2);
            // control_buf[3] is to control data transfer in MPE
            control_buf[3] = 1;
            end = false;
            while (!end) {
                control_buf[0] = 1;
                MPI_Send(control_buf, 4, MPI_INT, 0, 0, MPI_COMM_WORLD);
                // receive WID
                MPI_Recv(count_buf, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (count_buf[1] == 1) {
                    if (control_buf[2] == 1) {
                        for (int k=0; k<pe_per_node; k++) {
                            int tmpid = wid + k;
                            int localC, localD;
                            void * tptr = CollectiveInitialize(&localC, &localD, numParticles, tmpid, max_pe, false);
                            if (tmpid < max_pe) {
                                // send result data to MPE
                                MPI_Send(plDataset->mpParticle+localD, localC, mpiParticleType, 0, 0, MPI_COMM_WORLD);
                            }
                            control_buf[2] = 0;
                        }

                    }
                    wid = count_buf[2]; 

                    // Initialize dataset at beginning once
                    if (iter_switch) {
                        MPI_Recv(pDataset->mpParticle, numParticles,
                            mpiParticleType, MPI_NN_ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
                        iter_switch = false;
                        control_buf[3] = 0;
                    }

                    snprintf(wid_str, 8, "%d", count_buf[2]);
                    char     * c_argv[] = {const_cast<char *>("worker"), rank_str, wid_str, pConfigFile, NULL};
                    MPI_Comm   children_comm;
                    //Each of the processes in the master-job spawns a worker-job
                    //consisting of NUM_WORKER_PROCS processes.
                    int pe_spn;
                    if (wid+pe_per_node > max_pe)
                        pe_spn = max_pe - wid;
                    else
                        pe_spn = pe_per_node;
                    // host_info
                    char hostname[256];
                    gethostname(hostname, 256);
#if 1
                    int offset=0;
                    for (int i=0;i<strlen(hostname);i++) {
                        if (hostname[i] == '.') {
                            offset = i;
                            break;
                        }
                    }
                    hostname[offset] = '\0';
#endif
                    MPI_Info spawn_info;
                    MPI_Info_create(&spawn_info);
                    MPI_Info_set(spawn_info, "host", hostname);
                    // spawn children process
                    MPI_Comm_spawn(argv[0], c_argv, pe_spn, spawn_info/*MPI_INFO_NULL*/,
                        0, MPI_COMM_SELF, &children_comm, MPI_ERRCODES_IGNORE);

                    // send dataset and PE info
                    compute_info[0] = numParticles;
                    compute_info[1] = max_pe;
                    for (int k=0; k<pe_per_node; k++) {
                        if (wid+k < max_pe) {
                            MPI_Send(compute_info, 2, MPI_INT, k, 0, children_comm);
                            // send dataset to children PEs
                            MPI_Send(pDataset->mpParticle, pDataset->mNumParticles, mpiParticleType, k, 0, children_comm);
                        }
                    }
                    for (int k=0; k<pe_per_node; k++) { 
                        int tmpid = wid + k;
                        int localC, localD;
                        void * tptr = CollectiveInitialize(&localC, &localD, numParticles, tmpid, max_pe, false);

                        if (tmpid < max_pe) {
                            //Receive the message from all the corresponding workers.
                            MPI_Recv(plDataset->mpParticle+localD, localC, mpiParticleType, k, 0, children_comm, MPI_STATUS_IGNORE);
                        }
                    }

  
                    control_buf[1] = wid;
                    control_buf[2] = 1;

                } else if (count_buf[1] == 2) {
                    if (control_buf[2] == 1) {

                        for (int k=0; k<pe_per_node; k++) {
                            int tmpid = wid + k;
                            int localC, localD;
                            void * tptr = CollectiveInitialize(&localC, &localD, numParticles, tmpid, max_pe, false);
                            if (tmpid < max_pe) {
                                // Send result dataset
                                MPI_Send(plDataset->mpParticle+localD, localC, mpiParticleType, 0, 0, MPI_COMM_WORLD);
                            }
                            control_buf[2] = 0;
                        }

                    }
                    end = true;
                }
            
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
     gettimeofday(&t_e, NULL);
     t_p = (t_e.tv_sec + t_e.tv_usec * 1e-6) - (t_b.tv_sec + t_b.tv_usec * 1e-6);
     printf("Time: %f\n", t_p);
    }
    cout << "End of Program...... /" << rank << endl;
    FreeDatatype();
    MPI_Finalize();
    return 0;
}

/**
 * @Routine for the workers
 */

int WorkerRoutine(int argc, char ** argv) {

    MPI_Comm parent_comm;
    int      parent_size;
    int      task_buf[2];
    int      compute_info[2];

    MPI_Comm_get_parent(&parent_comm);
    if (parent_comm == MPI_COMM_NULL) 
        return -1;

    //Attention!: The size of the inter-communicator obtained through the
    //            MPI_Comm_remote_size() will always be '1' since a number
    //            of NUM_WORKER_PROCS child processes are spawned by each
    //            of the master processes. Therefore, each group of the
    //            NUM_WORKER_PROCS child processes recognizes only their
    //            correspodning master process in the inter-communicator.
    MPI_Comm_remote_size(parent_comm, &parent_size);

    // receive matrix info
    MPI_Recv(compute_info, 2, MPI_INT, 0, 0, parent_comm, MPI_STATUS_IGNORE);
    int numParticles = compute_info[0];
    max_pe = compute_info[1];
    wid = wid + rank;
    // vec_a & vec_b info
    pDataset = new ParticleDataset(numParticles);

    // receive dataset    
    MPI_Recv(pDataset->mpParticle, numParticles, mpiParticleType, 0, 0, parent_comm, MPI_STATUS_IGNORE);

    // computation initialize
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
    x##str1##Pos = pDataset->mpParticle[str2].xPos; \
    y##str1##Pos = pDataset->mpParticle[str2].yPos; \
    z##str1##Pos = pDataset->mpParticle[str2].zPos; \
    mass##str1 = pDataset->mpParticle[str2].mass

    ParticleDataset::Particle * localBuf;
    int                         localCnt;
    int                         localDisp;

    localBuf = (ParticleDataset::Particle *) CollectiveInitialize(&localCnt, &localDisp, numParticles, wid, max_pe, true);

    cout << "[" << wid << "/"
        << max_pe << "]: "
        << localCnt << " total: " << numParticles << endl;

    PRECISION step         = pConfig->mParams.timeRes;
    PRECISION duration     = pConfig->mParams.duration;
    PRECISION grav         = pConfig->mParams.gravConstant;
    pConfig->mParams.rank  = rank;

    printf("[%d]: %d/%d\n", wid, localDisp, localCnt);

    gConfig.localCnt  = localCnt;
    gConfig.localDisp = localDisp;

    char hostname[128];
    gethostname(hostname, 128);
    int deviceCnt = 0;
    CheckCUDAError(    cudaGetDeviceCount(&deviceCnt));
    //CheckCUDAError(    cudaSetDevice(rank % deviceCnt));
    cout << "\t\tPE:" << wid << "RUN: host[" << hostname << "]" << endl;

    /**********************************************************************
     * Launch Kernel
     *********************************************************************/

    CheckCUDAError(    CudaInitialize(pDataset, localBuf, pConfig, gConfig));
    CheckCUDAError(    ComputeParticleAttributes());
    CheckCUDAError(    CudaClean());

    CollectiveClean(localBuf);

#if 0
    struct timeval time_b, time_e;
    gettimeofday(&time_b, NULL);
    gettimeofday(&time_e, NULL);
    cout << "Kernel Time: " << (time_e.tv_usec - time_b.tv_usec)*1e-6 + ((double)time_e.tv_sec - (double)time_b.tv_sec) << endl;
#endif
    /**********************************************************************
     * Finalize
     *********************************************************************/
#if 0
    char send_buf[256];
    snprintf(send_buf, 256, "I am rank %d, the worker of rank %d, own rank: %d",
        wid, parent_rank, rank);
    printf( "Master(%d): %s : %d : %d\n", parent_rank, send_buf, mat_dim, pe_dim);
#endif
    task_buf[0] = 1;
    MPI_Send(pDataset->mpParticle+localDisp, localCnt, mpiParticleType, 0, 0, parent_comm);

    MPI_Finalize();

    return 0;
}

