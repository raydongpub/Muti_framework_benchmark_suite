#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
//#include <cuda.h>
#include <iostream>
#include <math.h>
//#include "cuda_runtime_api.h"
#include "commonBMT.h"
#include "cudaBMTKernel_MultiDim.cuh"

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
 * @File message
 */

#define MODE_NORM 0x0000
#define MODE_HELP 0x0001

#define DSET_XS   0x0000
#define DSET_S    0x0001
#define DSET_M    0x0002
#define DSET_L    0x0003
#define DSET_XL   0x0004

#define E_SUCCESS 0x0000
#define E_NO_ARG  0x0001
#define E_UNKNOWN 0x0002
#define E_INV_PE  0x0003
#define E_INV_PEV 0x0004
#define E_INV_DS  0x0005
#define E_INV_DSV 0x0006


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
int ret;
int iter;

/**
 * @Data type
 */
#ifndef _DOUBLE_PRECISION
#define PRECISION float
#else
#define PRECISION double
#endif
#ifdef _DOUBLE_PRECISION
#define MPI_PRECISION MPI_DOUBLE
#else
#define MPI_PRECISION MPI_FLOAT
#endif
PRECISION **** a,    **** b,    **** c,   *** p;
PRECISION  *** wrk1,  *** wrk2,  *** bnd;

Matrix       * pa, * pb, * pc, * pp,   * pwrk1, * pwrk2, *pbnd;
int            mx,   my,   mz,   imax,   jmax,    kmax,   it;
PRECISION      omega = 0.8;
PRECISION      wgosa, gosa;
/**
 * @himeno config
 */
int           gargc;
char       ** gargv;
int           mode = MODE_NORM;
BMT_Config    config;


//MPI
typedef struct {
    int l;
    int r;
} Neighbor;
typedef struct {
    Neighbor x;
    Neighbor y;
    Neighbor z;
} Cart_Neighbor;

int           numpes, peid, cartid[3];
MPI_Comm      comm_cart;
MPI_Datatype  jk_plane, ik_plane, ij_plane;
Cart_Neighbor nb;



#define NUM_WORKER_PROCS 1



/**
 * @Init parameter
 */

inline int SetPreference(int argc, char ** argv) {
#define IS_OPTION(str) (!strcmp(argv[idx], str))

    gargc = argc; gargv = argv;

    int idx = 1;

    while (idx < argc) {
        if IS_OPTION("-pe") {
            if ((idx + 3) >= argc)
                return E_INV_PE;
             int temp = atoi(argv[++idx]);
             if (temp < 1)
                 return E_INV_PEV;
             config.ndx0 = temp;

             temp = atoi(argv[++idx]);
             if (temp < 1)
                 return E_INV_PEV;
             config.ndy0 = temp;

             temp = atoi(argv[++idx]);
             if (temp < 1)
                 return E_INV_PEV;
             config.ndz0 = temp;
        }
        else if (IS_OPTION("-ds") || IS_OPTION("--dataset-size")) {
            if ((idx + 1) >= argc)
                return E_INV_DS;
            idx++;
            if IS_OPTION("xs") {
                config.mx0 = 33;
                config.my0 = 33;
                config.mz0 = 65;
            }
            else if IS_OPTION("s") {
                config.mx0 = 65;
                config.my0 = 65;
                config.mz0 = 129;
            }
            else if IS_OPTION("m") {
                config.mx0 = 129;
                config.my0 = 129;
                config.mz0 = 257;
            }
            else if IS_OPTION("l") {
                config.mx0 = 257;
                config.my0 = 257;
                config.mz0 = 513;
            }
            else if IS_OPTION("xl") {
                config.mx0 = 513;
                config.my0 = 513;
                config.mz0 = 1025;
            }
            else {
                return E_INV_DSV;
            }
        }
        else if (IS_OPTION("-h") || IS_OPTION("--help")) {
            mode = MODE_HELP;
            return E_SUCCESS;
        }
        idx++;
    }
    return E_SUCCESS;

#undef IS_OPTION
}





/**
 * @message functions
 */

const char * e_msg[] = {
"No error",
"No arguments specified.",
"Unrecognized arguments presented.",
"Requires three PE numbers along the dimensions.",
"Invalid PE numbers specified.",
"Requires the size of dataset (xs, s, m, l, xl).",
"Unrecognized dataset size"
};

const char h_msg[] = {"\
Usage: %s [OPTIONS] [...]                                                   \n\
Options are available in both short and long format:                      \n\n\
\t-pe [pe_x pe_y pe_z]     Specify numbers of PEs along dimensions          \n\
\t-h, --help               Show this help message.                          \n\
"};

/**
 * @Error check functions list
 */

inline void print_help() {
    char msg[512];
    sprintf(msg, h_msg, gargv[0]);
    cout << endl << msg << endl << endl;
}


inline void CheckError(int rc) {
    if (rc != E_SUCCESS) {
        cerr << endl << "Error: " << e_msg[rc] << endl;
        if (rc == E_NO_ARG)
            print_help();
        else
            cout << endl;
        exit(rc);
    }
}

inline void CheckCUDAError (cudaError_t ce) {
    if (ce != cudaSuccess) {
        cout << "CUDA_ERROR: " << cudaGetErrorString(ce) << endl;
        exit(0);
    }
}

/**
 * @Himeno initialize functions
 *
 */

//Work division and assignment for PEs
int bmtInitMax(int lmx, int lmy, int lmz, int peid) {

    int * mx1, * my1, * mz1;
    int * mx2, * my2, * mz2;
    int   tmp;

    mx1 = new int [config.mx0 + 1];
    my1 = new int [config.my0 + 1];
    mz1 = new int [config.mz0 + 1];

    mx2 = new int [config.mx0 + 1];
    my2 = new int [config.my0 + 1];
    mz2 = new int [config.mz0 + 1];

    tmp = mx / config.ndx0;
    mx1[0] = 0;
    for (int i=1;i<=config.ndx0;i++) {
        if (i <= mx % config.ndx0)
            mx1[i] = mx1[i - 1] + tmp + 1;
        else
            mx1[i] = mx1[i - 1] + tmp;
    }
    tmp = my / config.ndy0;
    my1[0] = 0;
    for (int i=1;i<=config.ndy0;i++) {
        if (i <= my % config.ndy0)
            my1[i] = my1[i - 1] + tmp + 1;
        else
            my1[i] = my1[i - 1] + tmp;
    }

    tmp = mz / config.ndz0;
    mz1[0] = 0;
    for (int i=1;i<=config.ndz0;i++) {
        if (i <= mz % config.ndz0)
            mz1[i] = mz1[i - 1] + tmp + 1;
        else
            mz1[i] = mz1[i - 1] + tmp;
    }
 //************************************************************************
    for(int i=0;i<config.ndx0;i++) {
        mx2[i] = mx1[i+1] - mx1[i];
        if(i != 0)
            mx2[i] = mx2[i] + 1;
        if(i != config.ndx0-1)
            mx2[i] = mx2[i] + 1;
    }

    for(int i=0;i<config.ndy0;i++) {
        my2[i] = my1[i+1] - my1[i];
        if(i != 0)
            my2[i] = my2[i] + 1;
        if(i != config.ndy0-1)
            my2[i] = my2[i] + 1;
    }

    for(int i=0;i<config.ndz0;i++) {
        mz2[i] = mz1[i+1] - mz1[i];
        if(i != 0)
            mz2[i] = mz2[i] + 1;
        if(i != config.ndz0-1)
            mz2[i] = mz2[i] + 1;
    }

    //************************************************************************
    imax = mx2[0];
    jmax = my2[0];
    kmax = mz2[peid];

    delete [] mz2;
    delete [] my2;
    delete [] mx2;

    delete [] mz1;
    delete [] my1;
    delete [] mx1;
    
    return 0;
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
            iter        = atoi(argv[idx + 3]);
        } 
        if (!strcmp(argv[idx], "-pm") && op_mode == OP_MODE_MASTER) {
            max_pe  = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-pn") && op_mode == OP_MODE_MASTER) {
            pe_node = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-pp") && op_mode == OP_MODE_MASTER) {
            pe_per_node = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-it")) {
            iter = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-v") && op_mode == OP_MODE_MASTER) {
           verify   = true;
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

    char rank_str[8], wid_str[8], iter_str[8];
    snprintf(rank_str, 8, "%d", rank);

    // Control PE master PE rank = 0
    int count = 0;  
    int count_buf[3];   
    int child_buf[2];
    int control_buf[4];
    int compute_info[7];
    PRECISION gosa_sum[pe_per_node];
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


    // Initialize data info
    CheckError(    SetPreference(argc, argv)); 
    config.mimax = (config.ndx0 == 1) ?
                    config.mx0 : (config.mx0 / config.ndx0) + 3;
    config.mjmax = (config.ndy0 == 1) ?
                    config.my0 : (config.my0 / config.ndy0) + 3;
    config.mkmax = (config.ndz0 == 1) ?
                    config.mz0 : (config.mz0 / config.ndz0) + 3;

    mx = config.mx0 - 1; my = config.my0 - 1; mz = config.mz0 - 1;

    if (mode == MODE_HELP) {
        print_help();
        exit(0);   
    }


    memset(count_buf, 0, sizeof(count_buf));
    memset(child_buf, 0, sizeof(child_buf));
    memset(control_buf, 0, sizeof(control_buf));
    memset(compute_info, 0, sizeof(compute_info));

    if (rank == 0) {
        cout << "++++++++" << endl << "PE_per_node" << pe_per_node << endl
            << "PE_Max: " << max_pe << endl 
            << "PE_Node: " << pe_node << endl
            << "PE_iter: " << iter << endl
            << "++++++++" << endl;
    }


    if (rank == 0 ) {
// Control PE (MPE)
        int idx;
        gettimeofday(&t_b, NULL);
        // control parameter 
        for (idx=0; idx<iter; idx++) {
            count = 0;
            memset(count_buf, 0, sizeof(count_buf));
            memset(child_buf, 0, sizeof(child_buf));
            memset(control_buf, 0, sizeof(control_buf));
            memset(compute_info, 0, sizeof(compute_info));
            memset(gosa_sum, 0, sizeof(gosa_sum));
            wgosa = 0.0; 
            gosa  = 0.0;
            while (count < max_pe) {
                if (idx == 0)
                    MPI_Recv(control_buf, 4, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                else {
                    count_buf[1] = 1;
                    for (int j=1; j<pe_node; j++)
                        MPI_Send(count_buf, 3, MPI_INT, j, 0, MPI_COMM_WORLD);
                }
#if 0
            MPI_Win_fence(0, win);
                cout << "==============" << endl << "Show Status: " << local_buf[i] << endl;
                cout << "==============" << endl << endl;
#endif
                if (idx == 0) {
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

                    } 
                    wid = count;
                    count += pe_per_node;
                    control_buf[0] = 0;
                    control_buf[1] = 0;
                    control_buf[2] = 0;
                }
                if (idx > 0) 
                    break;
            }        
            // receive the final dataset before ending
            for (int i=1; i< pe_node; i++) {
                if (idx == 0) {
                    // receive control_buf to have final data WID
                    MPI_Recv(control_buf, 4, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                    count_buf[1] = 2;
                    // send end flag to WPEs
                    MPI_Send(count_buf, 3, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
                //Receive result data from WPEs
                MPI_Recv(&wgosa, 1, MPI_PRECISION, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                gosa += wgosa;
            }
            //cout << "secs: " << sec << "/" << duration << "\xd";
            cout << idx << ":" << gosa << endl;

        }

    } else {
        MPI_Comm   children_comm[18];
        int        wid_buf[8];
        int        comm_count = 0;
        PRECISION  tmp_gosa   = 0.0;
        int idx;
        for (idx=0; idx<iter; idx++) {
            memset(count_buf, 0, sizeof(count_buf));
            memset(child_buf, 0, sizeof(child_buf));
            memset(control_buf, 0, sizeof(control_buf));
            memset(compute_info, 0, sizeof(compute_info));
            memset(gosa_sum, 0, sizeof(gosa_sum));
            tmp_gosa = 0.0;
            end      = false;
            while (!end) {
                control_buf[0] = 1;
                if (idx == 0) {
                    // send request for WID
                    MPI_Send(control_buf, 4, MPI_INT, 0, 0, MPI_COMM_WORLD);
                    // receive WID
                    MPI_Recv(count_buf, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                else {
                    MPI_Recv(count_buf, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                if (count_buf[1] == 1) {
                    if (idx == 0) {
                        // set WID
                        wid = count_buf[2]; 
                        snprintf(wid_str, 8, "%d", count_buf[2]);
                        snprintf(iter_str, 8, "%d", iter);
                        char     * c_argv[] = {const_cast<char *>("worker"), rank_str, wid_str, iter_str, NULL};
                        //Each of the processes in the master-job spawns a worker-job
                        //consisting of NUM_WORKER_PROCS processes.
                        int pe_spn;
                        if (wid+pe_per_node > max_pe)
                            pe_spn = max_pe - wid;
                        else
                            pe_spn = pe_per_node;
                        // host info
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
printf("pe: %d, pe_spn: %d, wid: %d\n", rank, pe_spn, wid);
                        MPI_Comm_spawn(argv[0], c_argv, pe_spn, spawn_info/*MPI_INFO_NULL*/,
                            0, MPI_COMM_SELF, &children_comm[comm_count], MPI_ERRCODES_IGNORE);
                    } 
                        // send dataset and PE info
                        compute_info[0] = config.ndx0;
                        compute_info[1] = config.ndy0;
                        compute_info[2] = config.ndz0;
                        compute_info[3] = config.mx0;
                        compute_info[4] = config.my0;
                        compute_info[5] = config.mz0;
                        compute_info[6] = gosa;
                    if (idx == 0) {
                        // Enable children PE to begin new iteration
                        for (int k=0; k<pe_per_node; k++) {
                            if (wid+k < max_pe) {
                                // send dataset to children PEs
                                MPI_Send(compute_info, 7, MPI_INT, k, 0, children_comm[comm_count]);
                            }
                        }
                        wid_buf[comm_count] = wid;
                        for (int k=0; k<pe_per_node; k++) { 
                            int tmpid = wid + k;
                            if (tmpid < max_pe) {
                                //Receive the message from all the corresponding workers.
                                MPI_Recv(&wgosa, 1, MPI_PRECISION, k, 0, children_comm[comm_count], MPI_STATUS_IGNORE);
                                tmp_gosa += wgosa;
                            }
                        }
                        comm_count++;

  
                        control_buf[1] = wid;
                        control_buf[2] = 1;
                    } else {
                        for (int j=0; j<comm_count; j++) {
                            for (int k=0; k<pe_per_node; k++) {
                                int tmpid = wid_buf[j] + k;
                                if (tmpid < max_pe) {
                                    // send dataset to children PEs
                                    MPI_Send(compute_info, 7, MPI_INT, k, 0, children_comm[j]);
                                }
                            }
                        }
                        for (int j=0; j<comm_count; j++) {
                            for (int k=0; k<pe_per_node; k++) {
                                int tmpid = wid_buf[j] + k;
                                if (tmpid < max_pe) {
                                    //Receive the message from all the corresponding workers.
                                    MPI_Recv(&wgosa, 1, MPI_PRECISION, k, 0, children_comm[j], MPI_STATUS_IGNORE);
                                    tmp_gosa += wgosa;
                                }
                            }

                        }
                        count_buf[1] =2;
                    }

                } else if (count_buf[1] == 2) {
                    // Send result dataset
                    MPI_Send(&tmp_gosa, 1, MPI_PRECISION, 0, 0, MPI_COMM_WORLD);
                    control_buf[2] = 0;
                    end = true;
                }//elseif
          
printf("Process: %d...begin\n", rank);
                if (idx > 0) {
                    MPI_Send(&tmp_gosa, 1, MPI_PRECISION, 0, 0, MPI_COMM_WORLD);
                    end = true;
                }//endif
printf("Process: %d...\n", rank);
            }//while
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
     gettimeofday(&t_e, NULL);
     t_p = (t_e.tv_sec + t_e.tv_usec * 1e-6) - (t_b.tv_sec + t_b.tv_usec * 1e-6);
     printf("Time: %f\n", t_p);
    }
    cout << "End of Program...... /" << rank << endl;
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
    int      compute_info[7];

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
    printf("\tchildren pe: %d, iter: %d\n", rank, iter);
    int idx;
    for (idx=0; idx<iter; idx++) {
    // receive matrix info
        MPI_Recv(compute_info, 7, MPI_INT, 0, 0, parent_comm, MPI_STATUS_IGNORE);

        if (idx == 0) {

            config.ndx0 = compute_info[0];
            config.ndy0 = compute_info[1];
            config.ndz0 = compute_info[2];
            config.mx0  = compute_info[3];
            config.my0  = compute_info[4];
            config.mz0  = compute_info[5];

            config.mimax = (config.ndx0 == 1) ?
                            config.mx0 : (config.mx0 / config.ndx0) + 3;
            config.mjmax = (config.ndy0 == 1) ?
                            config.my0 : (config.my0 / config.ndy0) + 3;
            config.mkmax = (config.ndz0 == 1) ?
                            config.mz0 : (config.mz0 / config.ndz0) + 3;

            mx = config.mx0 - 1; my = config.my0 - 1; mz = config.mz0 - 1;

            wid = wid + rank;

        // data initialize

            bmtInitMax(mx, my, mz, wid);

            pa    = new Matrix(4, config.mimax, config.mjmax, config.mkmax);
            pb    = new Matrix(3, config.mimax, config.mjmax, config.mkmax);
            pc    = new Matrix(3, config.mimax, config.mjmax, config.mkmax);
            pp    = new Matrix(config.mimax, config.mjmax, config.mkmax);
            pwrk1 = new Matrix(config.mimax, config.mjmax, config.mkmax);
            pwrk2 = new Matrix(config.mimax, config.mjmax, config.mkmax);
            pbnd  = new Matrix(config.mimax, config.mjmax, config.mkmax);

            bmtInitMt(
                *pa,   *pb,    *pc,
                *pp,   *pwrk1, *pwrk2,
                *pbnd,  mx,     it,
                config.mimax,  config.mjmax, config.mkmax,
                imax,   jmax,  kmax);

            cudaError_t ce = bmtInitDeviceMemory(
                                 pa,   pb,    pc,
                                 pp,   pwrk1, pwrk2,
                                 pbnd, rank);
            if (ce != cudaSuccess)
                cerr << "Error: " << cudaGetErrorString(ce) << endl;

            a    = pa->GetPtr4D();
            b    = pb->GetPtr4D();
            c    = pc->GetPtr4D();
            p    = pp->GetPtr3D();
            wrk1 = pwrk1->GetPtr3D();
            wrk2 = pwrk2->GetPtr3D();
            bnd  = pbnd->GetPtr3D();
        }
    int deviceCnt = 0;
    CheckCUDAError(    cudaGetDeviceCount(&deviceCnt));
    //    CheckCUDAError(    cudaSetDevice(rank % deviceCnt));
    char hostname[128];
    gethostname(hostname, 128);
    cout << "\tPE:" << wid << " / " << parent_rank << " iter: " <<iter << " ["
        << hostname << "]: RUN: Device[" << rank%deviceCnt << "]" << endl;

        /**********************************************************************
        * Launch Kernel
        *********************************************************************/

        CheckCUDAError(    bmtCudaJacobi(&wgosa, pp, imax, jmax, kmax));


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
    printf( "Master(%d): %s \n", parent_rank, send_buf);
#endif
        task_buf[0] = 1;
        MPI_Send(&wgosa, 1, MPI_PRECISION, 0, 0, parent_comm);
    }
    MPI_Finalize();
    return 0;
}

