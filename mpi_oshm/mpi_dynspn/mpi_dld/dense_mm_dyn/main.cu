#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <iostream>

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
 * @Common MPI attributes
 */
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

#define MPI_NN_ROOT 0
#ifndef _DOUBLE_PRECISION
#define MPI_PRECISION MPI_FLOAT
#else //_DOUBLE_PRECISION
#define MPI_PRECISION MPI_DOUBLE
#endif //_DOUBLE_PRECISION

/**
 * @Process attributes
 */
PRECISION * vec_a,   * vec_b,   * vec_c,   * vec_c_cpu;
PRECISION * vec_a_d, * vec_b_d, * vec_c_d;

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

inline void matdisplay(PRECISION * mat, int m_size, int m_dim) {
    for (int i = 0; i < m_size; i++) {
        if (i % (m_dim) == 0)
            cout << endl;
            cout << mat[i] << "\t";
    }
    cout << endl;
}

void matinitialize(PRECISION * mat, int m_size) {
    int num = m_size * m_size;
    for (int i = 0; i < num; i++) {
        mat[i] = ((PRECISION) (rand()%9)) + 1.0;
    }
    cout << "Matrix Created successfully" << endl;
}

/**
 * @ Kernel Function
 */

__global__ void MatrixMul_Kernel(
    PRECISION * va, PRECISION * vb,
    PRECISION * vc, int vec_stp,
    int mat_size, int vecc_num, int height) {

    int gridsize = gridDim.x * blockDim.x;
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int start, row, col;
    PRECISION sum = 0.0;
    for (int i = 0; i < vecc_num; i+= gridsize) {
        start = i + vec_stp + idx;
        row   = start / height;
        col   = start % height;
        if (start < (vecc_num+vec_stp)) {
            for (int j = 0; j < mat_size; j++) {
                sum  += va[row * mat_size + j] *
                vb[col * mat_size + j];
            }
            vc[start] = sum;
            sum = 0.0;
        }
    }
    __syncthreads();

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
        } 
        if (!strcmp(argv[idx], "-pe") && op_mode == OP_MODE_MASTER) {
            pe_dim  = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-m") && op_mode == OP_MODE_MASTER) {
            mat_dim = atoi(argv[idx+1]);
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


    }
    if (!matched)
        op_mode = OP_MODE_MASTER;
}

int main(int argc, char ** argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    SetOperationMode(argc, argv);

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

    if (rank == 0) {
        cout << "++++++++" << endl << "Matrix_Dim: " << mat_dim << " PE_per_node" << pe_per_node << endl 
            << "PE_Dim: " << pe_dim << endl << "PE_Node: " << pe_node << endl
            << "++++++++" << endl;
    }

    
    // PE and matrix info
    max_pe   = pe_dim * pe_dim;
    mat_size = mat_dim * mat_dim;  

    char rank_str[8], wid_str[8];
    snprintf(rank_str, 8, "%d", rank);

    // Control PE master PE rank = 0
    int count = 0;  
    int count_buf[3];   
    int child_buf[2];
    int control_buf[3];
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
    memset(control_buf, 0, 3);
    memset(compute_info, 0, 2);
    if (rank == 0 ) {
    // Matrix initilize
        int chunk_size = mat_dim / pe_dim;
        vec_a = new PRECISION[mat_size];
        vec_b = new PRECISION[mat_size];
        vec_c = new PRECISION[mat_size];     
        PRECISION * tmp_c = new PRECISION[chunk_size * chunk_size];

        matinitialize(vec_a, mat_dim);
        matinitialize(vec_b, mat_dim);
        gettimeofday(&t_b, NULL);
        while (count < max_pe) {
//                MPI_Irecv(count_buf, 3, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
//cout << "flag0" << endl;
            MPI_Recv(control_buf, 3, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
//cout << "flag1" << endl;
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

//cout << "flag11" << endl;
                // send WID
                MPI_Send(count_buf, 3, MPI_INT, rev_id, 0, MPI_COMM_WORLD);
// receive vec_c
                if (control_buf[2] == 1) {


                    for (int k=0; k<pe_per_node; k++) {

                        /**********************************************************************
                         * Identify work portion
                         **********************************************************************/
                        int cid = control_buf[1]+k; 
                        if (cid < max_pe) {

                            int vec_c_size   = chunk_size * chunk_size;


                            MPI_Recv(tmp_c, vec_c_size, MPI_PRECISION, rev_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            for (int j=0; j<vec_c_size; j++) {
                                int vec_c_offset = (cid/pe_dim)*chunk_size*mat_dim + (cid%pe_dim)*chunk_size;
                                int row = j/chunk_size;
                                int col = j%chunk_size;
                                vec_c[row*mat_dim + vec_c_offset + col] = tmp_c[j];
                            }
                        }
                    }

                }

                for (int k=0; k<pe_per_node; k++) {
                    /**********************************************************************
                     * Identify work portion
                     **********************************************************************/
                     wid = count+k;
                     if (wid < max_pe) {
                     // vec_a
                         int vec_a_tag    = wid / pe_dim;
                         int vec_a_offset = vec_a_tag * chunk_size * mat_dim;
                     // vec_b
                         int vec_b_tag    = wid % pe_dim;
                         int vec_b_offset = vec_b_tag * chunk_size * mat_dim;
                         int vec_b_size   = chunk_size * mat_dim;
                         int vec_a_size   = chunk_size * mat_dim;
                         int vec_c_size   = chunk_size * chunk_size;

                // send vec_a
                         MPI_Send((vec_a + vec_a_offset), vec_a_size, MPI_PRECISION, rev_id, 0, MPI_COMM_WORLD);
                // send vec_b
                         MPI_Send((vec_b + vec_b_offset), vec_b_size, MPI_PRECISION, rev_id, 0, MPI_COMM_WORLD);
                    }
                }
//cout << "flag4" << endl;
                count += pe_per_node;
                control_buf[0] = 0;
                control_buf[1] = 0;
                control_buf[2] = 0;
                       
            }

        }        

        for (int i=1; i< pe_node; i++) {
            //MPI_Irecv(count_buf, 3, MPI_INT, i, 0, MPI_COMM_WORLD, &request[i]); 
            MPI_Recv(control_buf, 3, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
// receive vec_c

            count_buf[1] = 2;
            MPI_Send(count_buf, 3, MPI_INT, i, 0, MPI_COMM_WORLD);
                int vec_c_size   = chunk_size * chunk_size;
                for (int k=0; k<pe_per_node; k++) {
                    int cid = control_buf[1] + k;
                    if (cid < max_pe) {
                        MPI_Recv(tmp_c, vec_c_size, MPI_PRECISION, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        for (int j=0; j<vec_c_size; j++) {
                            int vec_c_offset = (cid/pe_dim)*chunk_size*mat_dim + (cid%pe_dim)*chunk_size;
                            int row = j/chunk_size;
                            int col = j%chunk_size;
                            vec_c[row*mat_dim + vec_c_offset + col] = tmp_c[j];
                        }
                    }
                }

        }
        vec_c_cpu = new PRECISION[mat_size];
        PRECISION sum=0;
        if (verify) {
            // cpu version dense_mm
            for (int i=0; i<mat_size; i++) {
                int row = i/mat_dim;
                int col = i%mat_dim;
                for (int j=0; j<mat_dim; j++) { 
                    sum += vec_a[row*mat_dim + j] * vec_b[col*mat_dim + j];
                }
                vec_c_cpu[i] = sum;
                sum = 0;
            }
            for (int i=0; i<mat_size; i++) {
                if (vec_c[i] != vec_c_cpu[i]) {
                    cout << "\t\tError: Computation Error: gpu/cpu: " << vec_c[i] << "/" << vec_c_cpu[i] << " at: " << i << endl;
                    matdisplay(vec_c, mat_size, mat_dim);
                    matdisplay(vec_c_cpu, mat_size, mat_dim);
                    exit(0);
                }
            }
            cout << "Verification Successs!" << endl;
        }

    } else {
        // matrix info for working master PE
        int chunk_size = mat_dim / pe_dim;
        int vec_c_size   = chunk_size * chunk_size;
        int vec_c_tsize  = pe_per_node * vec_c_size;
        int vec_a_size   = chunk_size * mat_dim;
        int vec_a_tsize  = pe_per_node * vec_a_size;
        int vec_b_size   = chunk_size * mat_dim;
        int vec_b_tsize  = pe_per_node * vec_b_size;
        vec_a = new PRECISION[vec_a_tsize];
        vec_b = new PRECISION[vec_b_tsize];
        vec_c = new PRECISION[vec_c_tsize];


        while (!end) {
            control_buf[0] = 1;
            MPI_Send(control_buf, 3, MPI_INT, 0, 0, MPI_COMM_WORLD);


            // receive WID
            MPI_Recv(count_buf, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (count_buf[1] == 1) {
                if (control_buf[2] == 1) {
                    for (int k=0; k<pe_per_node; k++) {
                        if (wid+k < max_pe) {
                            MPI_Send((vec_c + k*vec_c_size), vec_c_size, MPI_PRECISION, 0, 0, MPI_COMM_WORLD);
                        }
                        control_buf[2] = 0;
                    }

                }
                wid            = count_buf[2]; 

                for (int k=0; k<pe_per_node; k++) {
                    // vec_a & vec_b info
                    int wtid = wid + k;
                    if (wtid < max_pe) {
                    // receive vec_a
                        MPI_Recv((vec_a + k*vec_a_size), vec_a_size, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // receive vec_b
                        MPI_Recv((vec_b + k*vec_b_size), vec_b_size, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }

                snprintf(wid_str, 8, "%d", count_buf[2]);
                char     * c_argv[] = {const_cast<char *>("worker"), rank_str, wid_str, NULL};
                MPI_Comm   children_comm;
                //Each of the processes in the master-job spawns a worker-job
                //consisting of NUM_WORKER_PROCS processes.
                int pe_spn;
                if (wid+pe_per_node > max_pe)
                    pe_spn = max_pe - wid;
                else
                    pe_spn = pe_per_node;
//DEBUG_R
                char hostname[256];
                gethostname(hostname, 256);
                //hostname[24] = '\0';
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
                
#if 0
   int universe_size, flag, * universe_sizep;
   MPI_Attr_get(MPI_COMM_WORLD, MPI_UNIVERSE_SIZE,  
                &universe_sizep, &flag);  
   if (!flag) { 
        printf("This MPI does not support UNIVERSE_SIZE. How many\n\ 
processes total?"); 
        scanf("%d", &universe_size); 
   } else universe_size = *universe_sizep; 
#endif

printf("flag0!, host: %s, pe_spn:%d, arg:%s\n", hostname, pe_spn, argv[0]);
#if 0
                MPI_Comm_spawn(argv[0], c_argv, pe_spn, /*spawn_info*/MPI_INFO_NULL,
                0, MPI_COMM_SELF, &children_comm, MPI_ERRCODES_IGNORE);
#else
                MPI_Comm_spawn(argv[0], c_argv, pe_spn, spawn_info/*MPI_INFO_NULL*/,
                0, MPI_COMM_SELF, &children_comm, MPI_ERRCODES_IGNORE);
#endif
printf("flag1!\n");

                // send matrix and PE info
                compute_info[0] = mat_dim;
                compute_info[1] = pe_dim;
                for (int k=0; k<pe_per_node; k++) {
                    if (wid+k < max_pe) {
                        MPI_Send(compute_info, 2, MPI_INT, k, 0, children_comm);
                // send matrix data
                        MPI_Send((vec_a + k*vec_a_size), vec_a_size, MPI_PRECISION, k, 0, children_comm);
                        MPI_Send((vec_b + k*vec_b_size), vec_b_size, MPI_PRECISION, k, 0, children_comm);
                    }
                }
  //  cout << "End of Computing......: " << rank << "/" << wid << "/" << max_pe<< endl;
                for (int k=0; k<pe_per_node; k++) { 
                    if (wid+k < max_pe) {
                //Receive the message from all the corresponding workers.
                        MPI_Recv((vec_c + k*vec_c_size), vec_c_size, MPI_PRECISION, k, 0, children_comm, MPI_STATUS_IGNORE);
                    }
                }
//DEBUG_R
                MPI_Comm_free(&children_comm);
                control_buf[1] = wid;
                control_buf[2] = 1;

            } else if (count_buf[1] == 2) {
                if (control_buf[2] == 1) {

                    for (int k=0; k<pe_per_node; k++) {
                        if (wid+k < max_pe) {
                        
                            MPI_Send((vec_c + k*vec_c_size), vec_c_size, MPI_PRECISION, 0, 0, MPI_COMM_WORLD);
                        }
                        control_buf[2] = 0;
                    }
                }
                end = true;
            }
            
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
     gettimeofday(&t_e, NULL);
     t_p = (t_e.tv_sec + t_e.tv_usec * 1e-6) - (t_b.tv_sec + t_b.tv_usec * 1e-6);
     printf("Time: %f\n", t_p);
    }
    cout << "End of Computing...... /" << rank << endl;
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
//cout << "flag15" << endl;
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
    mat_dim = compute_info[0];
    pe_dim  = compute_info[1];
    wid = wid + rank;
    // vec_a & vec_b info
    int chunk_size   = mat_dim / pe_dim;
    int mat_size     = mat_dim * mat_dim;
    int vec_a_size   = chunk_size * mat_dim;
    size_t size_a    = vec_a_size * sizeof(PRECISION);
    int vec_b_size   = chunk_size * mat_dim;
    size_t size_b    = vec_b_size * sizeof(PRECISION);
    int vec_c_size   = chunk_size * chunk_size;
    size_t size_c    = vec_c_size * sizeof(PRECISION);
    
    vec_a = new PRECISION[vec_a_size];
    vec_b = new PRECISION[vec_b_size];
    vec_c = new PRECISION[vec_c_size];
    // receive vec_a 
    MPI_Recv(vec_a, vec_a_size, MPI_PRECISION, 0, 0, parent_comm, MPI_STATUS_IGNORE);
    // receive vec_b
    MPI_Recv(vec_b, vec_b_size, MPI_PRECISION, 0, 0, parent_comm, MPI_STATUS_IGNORE);

    char hostname[128];
    gethostname(hostname, 128);
    int deviceCnt = 0;
    CheckCUDAError(    cudaGetDeviceCount(&deviceCnt));
    CheckCUDAError(    cudaSetDevice(rank % deviceCnt));
    cout << "\t\tPE:" << wid << "["
        << hostname << "]: RUN: Device[" << rank%deviceCnt << "]" << endl;

    /**********************************************************************
     * Launch Kernel
     *********************************************************************/

    CheckCUDAError(    cudaMalloc((void **) &vec_a_d, size_a));
    CheckCUDAError(    cudaMalloc((void **) &vec_b_d, size_b));
    CheckCUDAError(    cudaMalloc((void **) &vec_c_d, size_c));
    CheckCUDAError(    cudaMemcpy(vec_a_d, vec_a, size_a,
        cudaMemcpyHostToDevice));
    CheckCUDAError(    cudaMemcpy(vec_b_d, vec_b, size_b,
        cudaMemcpyHostToDevice));
    CheckCUDAError(    cudaMemset(vec_c_d, 0, size_c));

    struct timeval time_b, time_e;
    gettimeofday(&time_b, NULL);

    MatrixMul_Kernel <<<16, 128>>> (vec_a_d, vec_b_d, vec_c_d, 0,
        mat_dim, vec_c_size, chunk_size);

    CheckCUDAError(    cudaDeviceSynchronize());
    gettimeofday(&time_e, NULL);
    cout << "Kernel time: " << (time_e.tv_usec - time_b.tv_usec)*1e-6 + ((double)time_e.tv_sec - (double)time_b.tv_sec) << endl;

    CheckCUDAError(    cudaMemcpy(vec_c, vec_c_d, size_c,
        cudaMemcpyDeviceToHost));
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
    MPI_Send(vec_c, vec_c_size, MPI_PRECISION, 0, 0, parent_comm);
//DEBUG_R
    MPI_Comm_free(&parent_comm);
    MPI_Finalize();

    return 0;
}

