#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include "cuda_runtime_api.h"
#include <sys/types.h>
#include <signal.h>
#include "global.h"
#include "nw_gpu.h"

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
int pe_per_node;
int wid;

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
 * @nw configuration
 */
int div_fac = 10;
extern char blosum62_cpu[24][24];
extern int tile_size;
char * sequence_set1[MAX_STREAM];
char * sequence_set2[MAX_STREAM];
unsigned int * pos1[MAX_STREAM];
unsigned int * pos2[MAX_STREAM];
unsigned int * pos_matrix[MAX_STREAM];
int pair_num[MAX_STREAM];

//parameters
int NumberOfProcessors;
int seq1_len, seq2_len, seq_size, pos_size;
int seq_len;

/**
 * @message functions
 */
void usage(int argc, char **argv) {
    fprintf(stderr, "\nUsage: %s [options]\n", argv[0]);
    fprintf(stderr, "\t[--length|-l <length> ] - x and y length(default:");
    fprintf(stderr,"%d)\n",config.length);
    fprintf(stderr, "\t[--penalty|-p <penalty>] - penalty (negative");
    fprintf(stderr,"integer, default: %d)\n",config.penalty);
    fprintf(stderr, "\t[--num_pair|-n <pair num>] - number of pairs per");
    fprintf(stderr,"stream (default: %d)\n",config.num_streams);
    fprintf(stderr, "\t[--device|-d <device num> ]- device ID (default:");
    fprintf(stderr,"%d)\n",config.device);
    fprintf(stderr, "\t[--kernel|-k <kernel type> ]- 0: diagonal 1: tile");
    fprintf(stderr,"(default: %d)\n",config.kernel);
    fprintf(stderr, "\t[--num_blocks|-b <blocks> ]- blocks number per grid");
    fprintf(stderr,"(default: %d)\n",config.num_blocks);
    fprintf(stderr, "\t[--num_threads|-t <threads> ]- threads number per");
    fprintf(stderr,"block (default: %d)\n",config.num_threads);
    fprintf(stderr, "\t[--repeat|-r <num> ]- repeating number (default:");
    fprintf(stderr,"%d)\n",config.repeat);
    fprintf(stderr, "\t[--debug]- 0: no validation 1: validation (default:");
    fprintf(stderr,"%d)\n",config.debug);
    fprintf(stderr, "\t[--help|-h]- help information\n");
    exit(1);
}

void print_config() {
    fprintf(stderr, "=============== Configuration ================\n");
    fprintf(stderr, "device = %d\n", config.device);
    fprintf(stderr, "kernel = %d\n", config.kernel);
    if ( config.kernel == 1 )
        fprintf(stderr, "tile size = %d\n", tile_size);
    fprintf(stderr, "stream number = %d\n", config.num_streams);
    for (int i=0; i<config.num_streams; ++i) {
        fprintf(stderr, "Case %d - ", i );
        fprintf(stderr, "sequence number = %d\n", config.num_pairs[i]);
    } // end for
    fprintf(stderr, "sequence length = %d\n", config.length);
    fprintf(stderr, "penalty = %d\n", config.penalty);
    fprintf(stderr, "block number = %d\n", config.num_blocks);
    fprintf(stderr, "thread number = %d\n", config.num_threads);
    if ( config.num_streams==0 ) {
        fprintf(stderr, "\nNot specify sequence length\n");
    } //end if
    fprintf(stderr, "repeat = %d\n", config.repeat);
    fprintf(stderr, "debug = %d\n", config.debug);
    printf("==============================================\n");
} // end print_config()

double gettime() {
        struct timeval t;
        gettimeofday(&t,NULL);
        return t.tv_sec+t.tv_usec*1e-6;
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

inline void cudaCheckError(int line, cudaError_t ce) {
    if (ce != cudaSuccess){
        printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
        exit(1);
    } // end if
}  //end checkerror()

/**
 * @nw functions
 */
void init_conf() {
    config.debug = false;
    config.device = 0;
    config.kernel = 0;
    config.num_blocks = 16;
    config.num_threads = 32;
    config.num_streams = 0;
    config.length = 1600;
    config.penalty = -10;
    config.repeat = 1;
    config.dataset = 50;
}

void init_device(int device) {
    cudaSetDevice(device);
    printf("%d device is set", device);
}

void validate_config() {
    if ( config.length % tile_size != 0 &&
        config.kernel == 1 ) {
        fprintf(stderr, "Tile kernel used.\nSequence length should be");
        fprintf(stderr, "multiple times of tile size %d\n", tile_size);
        exit(1);
    }   // end if 
} // end void validate_config()

int parse_arguments(int argc, char **argv) {

    int i = 1;
    if (argc<6) {
        usage(argc, argv);
        return 0;
    } // end if
    while(i<argc) {
        if(strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") ==
          0){
            i++;
            if (i==argc){
                fprintf(stderr,"device number missing.\n");
                return 0 ;
                } // end if
            config.device = atoi(argv[i]);
        } // end if 
        else if(strcmp(argv[i], "--debug") == 0){
            config.debug = 1;
        } // end else if
        else if(strcmp(argv[i], "-k") == 0 || strcmp(argv[i],
                 "--kernel") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"device number missing.\n");
                return 0 ;
            }
            config.kernel = atoi(argv[i]);
        } // end elseif
        else if(strcmp(argv[i], "-r") == 0 || strcmp(argv[i],
                 "--repeat") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"repeating number missing.\n");
                return 0 ;
            }
            config.repeat = atoi(argv[i]);
        } // end elseif
        else if(strcmp(argv[i], "-t") == 0 || strcmp(argv[i],
                 "--num_threads") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"thread number missing.\n");
                return 0 ;
            }
            config.num_threads = atoi(argv[i]);
        } // end elseif
        else if(strcmp(argv[i], "-ds") == 0 || strcmp(argv[i],
                        "--dataset") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"Dataset size.\n");
                return 0 ;
            }
            config.dataset = atoi(argv[i]);
        } // end elseif
        else if(strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--num_blocks") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"block number missing.\n");
                return 0 ;
            }
            config.num_blocks = atoi(argv[i]);
        } // end elseif
        else if(strcmp(argv[i], "-p") == 0 || strcmp(argv[i],
                 "--penalty") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"penalty score missing.\n");
                return 0 ;
            }
            config.penalty = atoi(argv[i]);
        } //end elseif
        else if(strcmp(argv[i], "-n") == 0 || strcmp(argv[i],
                "--num_pairs") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"sequence length missing.\n");
                return 0 ;
            }
            config.num_pairs[ config.num_streams ] = atoi(argv[i]);
            if ( config.num_pairs[ config.num_streams ] >
                        MAX_SEQ_NUM ) {
                fprintf(stderr, "The maximum sequence number");
                fprintf(stderr,"per stream is %d\n", MAX_SEQ_NUM);
                return 0;
            } // end if  
            config.num_streams++;
        } //end elseif
        else if(strcmp(argv[i], "-l") == 0 || strcmp(argv[i],
                 "--lengths") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"sequence length missing.\n");
                return 0 ;
            } // end if 
            config.length = atoi(argv[i]);
            if ( config.length > MAX_SEQ_LEN ) {
                fprintf(stderr,"The maximum seqence length is");
                                fprintf(stderr,"%d\n", MAX_SEQ_LEN);
                return 0;
            } // end if 
        }  // end elseif
        else if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i],
                "--help") == 0) {
            usage(argc, argv);
            return 0;
        } //end else if 
        i++;
    }  // end while
    return 1;
}  // end parse arguments

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
            max_pe  = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-pn") && op_mode == OP_MODE_MASTER) {
            pe_node = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-pp") && op_mode == OP_MODE_MASTER) {
            pe_per_node = atoi(argv[idx+1]);
        }
        if (!strcmp(argv[idx], "-div") && op_mode == OP_MODE_MASTER) {
            div_fac = atoi(argv[idx+1]);
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

    init_conf();

    int ret;
    switch(op_mode) {
    case OP_MODE_MASTER:
        ret = MasterRoutine(argc, argv);
        break;
    case OP_MODE_WORKER:
        ret = WorkerRoutine(argc, argv);
        break;
    }

    MPI_Finalize();

    return 0;
}

/**
 * @Routine for the master:
 */

int MasterRoutine(int argc, char ** argv) {

    MPI_Barrier(MPI_COMM_WORLD);

    // init configuration of dataset
    while(!parse_arguments(argc, argv)) usage(argc, argv);
    seq_len=config.length;


    char rank_str[8], wid_str[8];
    snprintf(rank_str, 8, "%d", rank);

    // parameters
    NumberOfProcessors = max_pe;
    validate_config();
    int penalty = config.penalty;
     
    // Control PE master PE rank = 0
    int count = 0;  
    int count_buf[3];   
    int child_buf[2];
    int control_buf[4];
    int compute_info[4];
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
    memset(compute_info, 0, 4);

    if (rank == 0 ) {
// intialize dataset info
        gettimeofday(&t_b, NULL);
        srand(7);
        pair_num[0] = config.num_pairs[0];
        // data size define
        seq_size = pair_num[0] * seq_len;
        pos_size = pair_num[0];
        // init dataset
        for (int k=0; k<config.num_streams;++k) {
            sequence_set1[k] = new char[seq_size];
            sequence_set2[k] = new char[seq_size];
            pos1[k] = new unsigned int[pos_size];
            pos2[k] = new unsigned int[pos_size];
            pos_matrix[k] = new unsigned int[pos_size];
            pos_matrix[k][0] = pos1[k][0] = pos2[k][0] = 0;
            for (int i=0; i<pair_num[k]; ++i){
                //please define your own sequence 1
                seq1_len = seq_len; //64+rand() % 20;
                //printf("Seq1 length: %d\n", seq1_len);
                for (int j=0; j<seq1_len; ++j)
                    sequence_set1[k][ pos1[k][i] + j ] = rand() % 20 + 1;
                pos1[k][i+1] = pos1[k][i] + seq1_len;
                //please define your own sequence 2.
                seq2_len = seq_len;//64+rand() % 20;
                //printf("Seq2 length: %d\n\n", seq2_len);
                for (int j=0; j<seq2_len; ++j)
                    sequence_set2[k][ pos2[k][i] +j ] = rand() % 20 + 1;
                pos2[k][i+1] = pos2[k][i] + seq2_len;
                pos_matrix[k][i+1] = pos_matrix[k][i] + (seq1_len+1) * (seq2_len+1);
                dim_matrix[k][i] = (unsigned int)ceil((float)seq_len/tile_size )*tile_size;
            } // end for
        }// end for

// Control PE (MPE)
        // control parameter 
        bool iter_switch = true;
        control_buf[3] = 1;
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
                /**********************************************************************
                 * Identify work portion
                 **********************************************************************/
                wid = count;
                if (control_buf[3] == 1) {
                    MPI_Send(sequence_set1[0], seq_size, MPI_CHAR, rev_id, 0, MPI_COMM_WORLD); 
                    MPI_Send(sequence_set2[0], seq_size, MPI_CHAR, rev_id, 0, MPI_COMM_WORLD); 
                    MPI_Send(pos1[0], pos_size, MPI_UNSIGNED, rev_id, 0, MPI_COMM_WORLD); 
                    MPI_Send(pos2[0], pos_size, MPI_UNSIGNED, rev_id, 0, MPI_COMM_WORLD); 
                    MPI_Send(pos_matrix[0], pos_size, MPI_UNSIGNED, rev_id, 0, MPI_COMM_WORLD); 
                    control_buf[3] =0;    
                }
                count += pe_per_node;
                control_buf[0] = 0;
                control_buf[1] = 0;
                control_buf[2] = 0;
                       
            }

        }//endwhile        
            // receive the final dataset before ending
        for (int i=1; i< pe_node; i++) {
            // receive control_buf to have final data WID
            MPI_Recv(control_buf, 4, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            count_buf[1] = 2;
            // send end flag to WPEs
            MPI_Send(count_buf, 3, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // dataset info
        pair_num[0] = config.num_pairs[0];
        // data size define
        seq_size = pair_num[0] * seq_len;
        pos_size = pair_num[0];
        // init dataset
        for (int k=0; k<config.num_streams;++k) {
            sequence_set1[k] = new char[seq_size];
            sequence_set2[k] = new char[seq_size];
            pos1[k] = new unsigned int[pos_size];
            pos2[k] = new unsigned int[pos_size];
            pos_matrix[k] = new unsigned int[pos_size];

        }
        bool iter_switch = true;
            // control_buf[3] is to control data transfer in MPE
        end = false;
        control_buf[3] = 1;
        while (!end) {
            control_buf[0] = 1;
            MPI_Send(control_buf, 4, MPI_INT, 0, 0, MPI_COMM_WORLD);
            // receive WID
            MPI_Recv(count_buf, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (count_buf[1] == 1) {
                // set WID
                wid = count_buf[2]; 
                // Initialize dataset at beginning once
                if (iter_switch) {
                    MPI_Recv(sequence_set1[0], seq_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(sequence_set2[0], seq_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(pos1[0], pos_size, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(pos2[0], pos_size, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(pos_matrix[0], pos_size, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    control_buf[3] = 0;
                    iter_switch = false;
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
                // spawn children process
                MPI_Comm_spawn(argv[0], c_argv, pe_spn, spawn_info/*MPI_INFO_NULL*/,
                    0, MPI_COMM_SELF, &children_comm, MPI_ERRCODES_IGNORE);
                // send dataset and PE info
                compute_info[0] = max_pe;
                compute_info[1] = seq_len;
                compute_info[2] = pair_num[0]; 
                compute_info[3] = div_fac;
                for (int k=0; k<pe_per_node; k++) {
                    if (wid+k < max_pe) {
                        MPI_Send(compute_info, 4, MPI_INT, k, 0, children_comm);
                        // send dataset to children PEs
                        MPI_Send(sequence_set1[0], seq_size, MPI_CHAR, k, 0, children_comm);
                        MPI_Send(sequence_set2[0], seq_size, MPI_CHAR, k, 0, children_comm);
                        MPI_Send(pos1[0], pos_size, MPI_UNSIGNED, k, 0, children_comm);
                        MPI_Send(pos2[0], pos_size, MPI_UNSIGNED, k, 0, children_comm);
                        MPI_Send(pos_matrix[0], pos_size, MPI_UNSIGNED, k, 0, children_comm);
                    }
                }
                for (int k=0; k<pe_per_node; k++) { 
                    int tmpid = wid + k;
                    if (tmpid < max_pe) {
                        //Receive the message from all the corresponding workers.
                        MPI_Recv(child_buf, 2, MPI_INT, k, 0, children_comm, MPI_STATUS_IGNORE);
                    }
                }

                control_buf[1] = wid;
                control_buf[2] = 1;
            } else if (count_buf[1] == 2) {
                end = true;
            }//end elseif
        }//end while
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
     gettimeofday(&t_e, NULL);
     t_p = (t_e.tv_sec + t_e.tv_usec * 1e-6) - (t_b.tv_sec + t_b.tv_usec * 1e-6);
     printf("Time: %f\n", t_p);
    }
    cout << "End of Program...... /" << rank << endl;
    //MPI_Finalize();
    return 0;
}

/**
 * @Routine for the workers
 */

int WorkerRoutine(int argc, char ** argv) {

    MPI_Comm parent_comm;
    int      parent_size;
    int      task_buf[2];
    int      compute_info[4];
    cudaStream_t stream[1];

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
    // define time stamp
    double s_time, e_time, p_time,
           sd_time, ed_time, pd_time,
           se_time, ee_time, pe_time;
    double b_time, f_time, bf_time;


    // receive matrix info
    MPI_Recv(compute_info, 4, MPI_INT, 0, 0, parent_comm, MPI_STATUS_IGNORE);
    // init parameter
    int NumberOfProcessors = compute_info[0];
    max_pe      = compute_info[0];
    seq_len     = compute_info[1];
    pair_num[0] = compute_info[2];
    int *score_matrix[1];
    int *full_score_matrix[1];
   div_fac     = compute_info[3];
    //init dataset info
    seq_size = pair_num[0] * seq_len;
    pos_size = pair_num[0];
    // init dataset
    for (int k=0; k<1;++k) {
        sequence_set1[k] = new char[seq_size];
        sequence_set2[k] = new char[seq_size];
        pos1[k] = new unsigned int[pos_size];
        pos2[k] = new unsigned int[pos_size];
        pos_matrix[k] = new unsigned int[pos_size];
    }
    wid = wid + rank;

    // receive dataset    
    MPI_Recv(sequence_set1[0], seq_size, MPI_CHAR, 0, 0, parent_comm, MPI_STATUS_IGNORE);
    MPI_Recv(sequence_set2[0], seq_size, MPI_CHAR, 0, 0, parent_comm, MPI_STATUS_IGNORE);
    MPI_Recv(pos1[0], pos_size, MPI_UNSIGNED, 0, 0, parent_comm, MPI_STATUS_IGNORE);
    MPI_Recv(pos2[0], pos_size, MPI_UNSIGNED, 0, 0, parent_comm, MPI_STATUS_IGNORE);
    MPI_Recv(pos_matrix[0], pos_size, MPI_UNSIGNED, 0, 0, parent_comm, MPI_STATUS_IGNORE);

/**
*** Partition work begin
**/
    int pair_num_pe  = pair_num[0];
    int score_num_pe = pair_num_pe*(seq_len+1)*(seq_len+1);
    pair_num[0]     /= div_fac;
    int pair_fix_num = pair_num[0];
    int offset       = 0;
    int data_offset  = pair_num_pe * seq_len * (wid) / NumberOfProcessors;
    full_score_matrix[0] = (int *)malloc(pair_num_pe*(seq_len+1)*(seq_len+1));
    p_time  = 0.0;
    pd_time = 0.0;
    pe_time = 0.0;
    for (int k=0;k<1; k++){
        for (int i=0; i<=pair_num[k]/NumberOfProcessors; i++){
            if (i==0) {
                pos1[k][i]=0;
                pos2[k][i]=0;
                pos_matrix[k][i]=0;
            }  // end if 
            else {
                pos1[k][i]       = pos1[k][i-1] + seq_len;
                pos2[k][i]       = pos2[k][i-1] + seq_len;
                pos_matrix[k][i] = pos_matrix[k][i-1] + (seq_len+1)*(seq_len+1);
            } // end else
        } // end for
    } // end for


    for (int k=0;k<1; k++) {
        pair_num[k] /= NumberOfProcessors;
        score_matrix[k] = (int *)malloc(pos_matrix[k][pair_num[k]]*sizeof(int) );
    } // end for
            //printf("\n\t\tKernel:%d The pair_num:%d, The pair_size: %u\n", config.kernel, pair_num[0], sizeof(int)*pos_matrix[0][pair_num[0]]);



    // computation initialize
    s_time = gettime();

    char hostname[128];
    gethostname(hostname, 128);
    int deviceCnt = 0;
    cudaCheckError( __LINE__, cudaGetDeviceCount(&deviceCnt));
    int dev_id = rank % deviceCnt;
    cudaCheckError( __LINE__, cudaSetDevice(dev_id));
    printf("\t\tPE:[%d], RUN: host %s, device[%d]\n", rank, hostname, dev_id);
    printf( "****************** DEVID:  %d ******************\n\n", dev_id);
    config.device = dev_id;
    //print_config();
    e_time = gettime();
    p_time += (e_time-s_time);
    sd_time = gettime();
    for (int i=0; i<1; ++i) {
        //nw_gpu_allocate(i);
        //Added GPU alloc here:
        //int i = stream_num;
        cudaCheckError( __LINE__,
            cudaMalloc( (void**)&d_sequence_set1[i],
            sizeof(char)*pos1[i][pair_num[i]] ) );
        cudaCheckError( __LINE__,
            cudaMalloc( (void**)&d_sequence_set2[i],
            sizeof(char)*pos2[i][pair_num[i]] ) );
        cudaCheckError( __LINE__,
            cudaMalloc( (void**)&d_score_matrix[i],
            sizeof(int)*pos_matrix[i][pair_num[i]]) );
        cudaCheckError( __LINE__,
            cudaMalloc( (void**)&d_pos1[i],
            sizeof(unsigned int)*(pair_num[i]+1) ) );
        cudaCheckError( __LINE__,
            cudaMalloc( (void**)&d_pos2[i],
            sizeof(unsigned int)*(pair_num[i]+1) ) );
        cudaCheckError( __LINE__,
            cudaMalloc( (void**)&d_pos_matrix[i],
            sizeof(unsigned int)*(pair_num[i]+1) ) );
        cudaCheckError( __LINE__,
            cudaMalloc( (void**)&d_dim_matrix[i],
            sizeof(unsigned int)*(pair_num[i]+1) ) );
    }//end for
    //record cudaMalloc time
    ed_time  = gettime();
    pd_time += ed_time - sd_time;

    for (int pt=0; pt<div_fac;pt++) {

#ifdef _PRINTOUT
        printf("****** Iteration:%d ******\n", pt);
#endif
        /* initialize the GPU */
        offset       = pt * pair_num[0] * seq_len;
//printf("*** Rank: %d, data_pe: %d, offset is: %d, data_offset: %d ***\n", Rank, pos1[0][pair_num[0]], offset, data_offset);
        for (int i=0; i<1; ++i) {
            /* Memcpy to device */
            // record cudaMemcpy & kernel 
            se_time = gettime();
            cudaCheckError( __LINE__,
                cudaMemcpy( d_sequence_set1[i],
                &sequence_set1[i][offset+data_offset],
                sizeof(char)*pos1[i][pair_num[i]],
                cudaMemcpyHostToDevice ) );
            cudaCheckError( __LINE__,
                cudaMemcpy( d_sequence_set2[i],
                &sequence_set2[i][offset+data_offset],
                sizeof(char)*pos2[i][pair_num[i]],
                cudaMemcpyHostToDevice ) );
            cudaCheckError( __LINE__,
                cudaMemcpy( d_pos1[i], pos1[i],
                sizeof(unsigned int)*(pair_num[i]+1),
                cudaMemcpyHostToDevice ) );
            cudaCheckError( __LINE__,
                cudaMemcpy( d_pos2[i], pos2[i],
                sizeof(unsigned int)*(pair_num[i]+1),
                cudaMemcpyHostToDevice ) );
            cudaCheckError( __LINE__,
                cudaMemcpy( d_pos_matrix[i],
                pos_matrix[i],
                sizeof(unsigned int)*(pair_num[i]+1),
                cudaMemcpyHostToDevice ) );
            cudaCheckError( __LINE__,
                cudaMemcpy( d_dim_matrix[i], dim_matrix[i],
                sizeof(unsigned int)*(pair_num[i]+1),
                cudaMemcpyHostToDevice ) );
            cudaCheckError( __LINE__,
                cudaStreamCreate( &(stream[i]) ));
        } // end for




        /**********************************************************************
         * Launch Kernel
         *********************************************************************/
        for (int i=0; i<1; ++i ) {
            nw_gpu(sequence_set1[i],
                   sequence_set2[i],
                   pos1[i], pos2[i],
                   score_matrix[i], pos_matrix[i],
                   pair_num[i],d_score_matrix[i],
                   stream[0], i, 0);
#ifdef _PRINTOUT
            printf("Kernel called\n");
#endif
/* Remove the data copy back
                if (Rank==0) {
                    nw_gpu_copyback(score_matrix[i], d_score_matrix[i],
                    pos_matrix_rank[i], pair_num[i], stream[0],i);
                    for (int ic=0; 
                        ic<pos_matrix_rank[i][pair_num[i]]; 
                        ++ic) {
                        full_score_matrix[i][ic+score_offset] = 
                            score_matrix[i][ic];

                    }
                }*/
            cudaCheckError( __LINE__,
                cudaStreamSynchronize(stream[0]));

        } // end for
        ee_time  = gettime();
        pe_time += ee_time-se_time;

#ifdef _PRINTOUT
        printf("****** Iteration:%d end ******\n", pt);
#endif
    } // end for
    f_time = gettime();
    bf_time = f_time - b_time;
    printf("++++++++++COMPLETE+++++++++++++++\n");




#if 0
    struct timeval time_b, time_e;
    gettimeofday(&time_b, NULL);
    gettimeofday(&time_e, NULL);
    cout << "Kernel Time: " << (time_e.tv_usec - time_b.tv_usec)*1e-6 + ((double)time_e.tv_sec - (double)time_b.tv_sec) << endl;
#endif
    /**********************************************************************
     * Finalize
     *********************************************************************/
    char send_buf[256];
    snprintf(send_buf, 256, "I am rank %d, the worker of rank %d, own rank: %d",
        wid, parent_rank, rank);
    printf( "Master(%d): %s\n", parent_rank, send_buf);
    task_buf[0] = 1;
    MPI_Send(task_buf, 2, MPI_INT, 0, 0, parent_comm);

    return 0;
}

