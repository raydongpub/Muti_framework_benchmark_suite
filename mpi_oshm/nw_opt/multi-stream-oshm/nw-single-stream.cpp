    /*
    * File:  nw-multi-stream.cu
    * Author: Da Li
    * Email:  da.li@mail.missouri.edu
    * Organization: Networking and Parallel Systems Lab
    * (http://nps.missouri.edu/)
    *
    * Description: This file is the framework of program.
    *
    */

    #include <stdio.h>
    #include <stdlib.h>
    #include <cuda_runtime_api.h>
    #include "global.h"
    #include "nw_cpu.h"
    #include "nw_gpu.h"
    #include <shmem.h>

    #include <sys/types.h>
    #include <sys/time.h>
    #include <unistd.h>
    #include <signal.h>
    #include <string.h>


    extern char blosum62_cpu[24][24];
    extern int tile_size;

    void init_conf() {
    config.debug = false;
    config.device = 0;
    config.kernel = 0;
    config.num_blocks = 32;
    config.num_threads = 512;
    config.num_streams = 0;
    config.length = 1600;
    config.penalty = -10;
    config.repeat = 1;
    config.dataset = 50;
    config.div_fac = 5;
    }

    void init_device(int device) {
    cudaSetDevice(device);
//    printf("%d device is set", device);
    }



    inline void cudaCheckError(int line, cudaError_t ce)
    {
        if (ce != cudaSuccess){
        printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
        shmem_finalize();
        exit(1);
    } // end if
    }  //end checkerror()

    void usage(int argc, char **argv)
    {
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
    shmem_finalize();
    exit(1);
    }

    void print_config()
    {
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
    fprintf(stderr, "div_fac = %d\n", config.div_fac);
    printf("==============================================\n");
    } // end print_config()

    void validate_config()
    {
    if ( config.length % tile_size != 0 &&
        config.kernel == 1 ) {
        fprintf(stderr, "Tile kernel used.\nSequence length should be");
        fprintf(stderr, "multiple times of tile size %d\n", tile_size);
        shmem_finalize();
        exit(1);
    }   // end if 
    } // end void validate_config()

    int validation(int *score_matrix_cpu, int *score_matrix, unsigned int
    length)
    {
    unsigned int i = 0;
    //printf("Length : %d\n", length);
    while (i!=length){
        if ( score_matrix_cpu[i]==score_matrix[i] ){
            //printf("On GPU: score_matrix[%d] = %d\n", i,
            //score_matrix[i]);
            //printf("On CPU: score_matrix[%d] = %d\n", i,
            //score_matrix_cpu[i]);
            ++i;
            continue;
        } // end if 
        else {
            printf("On GPU: score_matrix[%d] = %d\n", i, score_matrix[i]);
            printf("On CPU: score_matrix[%d] = %d\n", i,score_matrix_cpu[i]);
           //++i;
            return 0;
        } // end else
    } // end while
    return 1;
    }  // end validation()
#if 0
    void sendtosingleprocess(int *score_matrix, unsigned int length, int Rank,
    int NumberOfProcessors){
        MPI_Status status;
        unsigned int i=0;
        int *central_node = (int
        *)malloc(NumberOfProcessors*pos_matrix[i][pair_num[i]]*sizeof(int));

        MPI_Gather(score_matrix, length, MPI_INT, central_node, length,
        MPI_INT, 0, MPI_COMM_WORLD);
        free(central_node);

    } // end sendtosingleprocess()
#endif

    int parse_arguments(int argc, char **argv)
    {
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
        else if(strcmp(argv[i], "-f") == 0 || strcmp(argv[i],
                 "--div_fac") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"div_fac missing.\n");
                return 0 ;
            }
            config.div_fac = atoi(argv[i]);
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
        else if(strcmp(argv[i], "-b") == 0 || strcmp(argv[i],
                        "--num_blocks") == 0){
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
        else {
            fprintf(stderr,"Unrecognized option : %s\nTry --help\n", argv[i]);
            return 0;
        }
        i++;
    }  // end while
    return 1;
    }  // end parse arguments

    int main(int argc, char **argv)
    {
    shmem_init();
    double s_time, e_time, p_time, 
           sd_time, ed_time, pd_time,
           se_time, ee_time, pe_time;
    int Rank;
    int NumberOfProcessors;
    init_conf();
    while(!parse_arguments(argc, argv)) usage(argc, argv);
    int seq1_len, seq2_len; 
    int seq_len=config.length;
    int dataset=config.dataset;
    /*printf("Dataset is %d", dataset);
    printf("seq_length is %d", seq_len);*/
    Rank = shmem_my_pe();
    NumberOfProcessors = shmem_n_pes();
/**
*** add work partition factor
**/
    int div_fac = config.div_fac;
//    config.num_pairs[0] /= div_fac;  


    while(dataset>0){
        //print_config();
        validate_config();
        int dev_num = config.device;
        DEBUG = config.debug;
        int penalty = config.penalty;
        int *score_matrix[config.num_streams];
        int *full_score_matrix[config.num_streams];
        cudaStream_t stream[config.num_streams];
        if(Rank==0){
          print_config();
            srand ( 7 );
            for (int k=0; k<config.num_streams;++k) {
#if 0
                sequence_set1[k] = new char [pair_num[0] * seq_len];
                sequence_set2[k] = new char [pair_num[0] * seq_len];
                pos1[k] = new unsigned int [pair_num[0]];
                pos2[k] = new unsigned int [pair_num[0]];
                pos_matrix[k] = new unsigned int [pair_num[0]];
#endif

                pos_matrix[k][0] = pos1[k][0] = pos2[k][0] = 0;
                pair_num[k] = config.num_pairs[k];
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
                   pos_matrix[k][i+1] = pos_matrix[k][i] + (seq1_len+1) *
                   (seq2_len+1);
                   dim_matrix[k][i] = (unsigned int)ceil(
                   (float)seq_len/tile_size )*tile_size;
                } // end for
           } // end for
        }  // end if

        for (int k=0; k<config.num_streams;++k) {
            pair_num[k] = config.num_pairs[k];
        } // end for
        char sequence_set_rank[config.num_streams][5000*seq_len/                        NumberOfProcessors];
        char sequence_set_rank2[config.num_streams][5000*seq_len/                       NumberOfProcessors];
        unsigned int pos1_rank[config.num_streams][5000];
        unsigned int pos2_rank[config.num_streams][5000];
        unsigned int pos_matrix_rank[config.num_streams][5000];
        shmem_barrier_all();
        for (int k=0;k<config.num_streams;k++){
           if (pair_num[k]%NumberOfProcessors!=0){
               printf("Number of Processors not divisible by Number of Pairs");
               fprintf(stderr,"%d. Change number of pairs\n\n\n", pair_num[k]);
               shmem_finalize();
               exit(0);
           }// end if 

            else{
//                printf("Each gets %d\n",config.num_pairs[k]/NumberOfProcessors);
            } // end else
        } // enf for



        shmem_barrier_all();
        char * set1_s, *set2_s;
        unsigned int * pos1_s, * pos2_s, * pos_matrix_s;        
        int nump, nums, nump_c, nums_c;
#if 1
        for (int k=0; k<config.num_streams;k++){
            pair_num[k]=config.num_pairs[k];
            nump = pair_num[k];
            nums = nump * seq_len;
            nump_c = nump/NumberOfProcessors;
            nums_c = nums/NumberOfProcessors;
            // allocate communication buffer
            set1_s = (char *) shmem_malloc(nums*sizeof(char));
            set2_s = (char *) shmem_malloc(nums*sizeof(char));
            pos1_s = (unsigned int*) shmem_malloc(nump * sizeof(unsigned int));
            pos2_s = (unsigned int*) shmem_malloc(nump * sizeof(unsigned int));
            pos_matrix_s = (unsigned int*) shmem_malloc(nump * sizeof(unsigned int));
            if (Rank == 0) {
                memcpy(set1_s, sequence_set1[k], nums*sizeof(char));
                memcpy(set2_s, sequence_set2[k], nums*sizeof(char));
                memcpy(pos1_s, pos1[k], nump*sizeof(unsigned int));
                memcpy(pos2_s, pos2[k], nump*sizeof(unsigned int));
                memcpy(pos_matrix_s, pos_matrix[k], nump*sizeof(unsigned int));
            }
            shmem_barrier_all();
            if (Rank !=0) {
                shmem_getmem(set1_s, &set1_s[Rank*nums_c], nums_c*sizeof(char), 0);
                shmem_getmem(set2_s, &set2_s[Rank*nums_c], nums_c*sizeof(char), 0);
                shmem_getmem(pos1_s, &pos1_s[Rank*nump_c], nump_c*sizeof(unsigned int), 0);
                shmem_getmem(pos2_s, &pos2_s[Rank*nump_c], nump_c*sizeof(unsigned int), 0);
                shmem_getmem(pos_matrix_s, &pos_matrix_s[Rank*nump_c], nump_c*sizeof(unsigned int), 0);
            }
            memcpy(sequence_set_rank[k], set1_s, nums_c*sizeof(char));
            memcpy(sequence_set_rank2[k], set2_s, nums_c*sizeof(char));
            memcpy(pos1_rank[k], pos1_s, nump_c*sizeof(unsigned int));
            memcpy(pos2_rank[k], pos2_s,nump_c*sizeof(unsigned int));
            memcpy(pos_matrix_rank[k], pos_matrix_s, nump_c*sizeof(unsigned int));
            shmem_barrier_all();
            shmem_free(set1_s);
            shmem_free(set2_s);
            shmem_free(pos1_s);
            shmem_free(pos2_s);
            shmem_free(pos_matrix_s);
        } // end for
#endif
/**
*** Partition work begin
**/
    int pair_num_pe  = pair_num[0];
    int score_num_pe = pair_num_pe*(seq_len+1)*(seq_len+1);
    pair_num[0]     /= div_fac;
    int pair_fix_num = pair_num[0];
    int offset       = 0;
    int score_offset = 0;
    full_score_matrix[0] = (int *)malloc(pair_num_pe*(seq_len+1)*(seq_len+1));
    p_time  = 0.0;
    pd_time = 0.0;
    pe_time = 0.0;
        
        for (int k=0;k<config.num_streams; k++){
            for (int i=0;i<=pair_num[k]/NumberOfProcessors; i++){
                if (i==0){
                    pos1_rank[k][i]=0;
                    pos2_rank[k][i]=0;
                    pos_matrix_rank[k][i]=0;
                }  // end if 
                else{
                    pos1_rank[k][i]=pos1_rank[k][i-1]+seq_len;
                    pos2_rank[k][i]=pos2_rank[k][i-1]+seq_len;
                    pos_matrix_rank[k][i]=pos_matrix_rank[k][i-1]
                    + (seq_len+1)*(seq_len+1);
                } // end else
            } // end for
        } // end for


        for (int k=0;k<config.num_streams; k++) {
            pair_num[k] /=  NumberOfProcessors;
            score_matrix[k] = (int *)malloc(
            pos_matrix_rank[k][pair_num[k]]*sizeof(int) );
        } // end for
//        printf("\n\t\tKernel:%d The pair_num:%d, The pair_size: %u\n", config.kernel, pair_num[0], sizeof(int)*pos_matrix_rank[0][pair_num[0]]);
        s_time = gettime();
        int numdev; cudaGetDeviceCount(&numdev);
        int setdevice=Rank%numdev;
        init_device( setdevice );
        //init_device(0);
        e_time = gettime();
        p_time += (e_time-s_time);
        sd_time = gettime();
        for (int i=0; i<config.num_streams; ++i) {
            //nw_gpu_allocate(i);
            //Added GPU alloc here:
            //int i = stream_num;
            cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set1[i],
            sizeof(char)*pos1_rank[i][pair_num[i]] ) );
        
            cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sequence_set2[i],
            sizeof(char)*pos2_rank[i][pair_num[i]] ) );
        
            cudaCheckError( __LINE__, cudaMalloc( (void**)&d_score_matrix[i],
            sizeof(int)*pos_matrix_rank[i][pair_num[i]]) );
        
            cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos1[i],
            sizeof(unsigned int)*(pair_num[i]+1) ) );
        
            cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos2[i],
            sizeof(unsigned int)*(pair_num[i]+1) ) );
        
            cudaCheckError( __LINE__, cudaMalloc( (void**)&d_pos_matrix[i],
            sizeof(unsigned int)*(pair_num[i]+1) ) );    
            
            cudaCheckError( __LINE__, cudaMalloc( (void**)&d_dim_matrix[i],                 sizeof(unsigned int)*(pair_num[i]+1) ) );
        }
        ed_time  = gettime();
        pd_time += ed_time - sd_time;

    for (int pt=0; pt<div_fac;pt++) {
        printf("****** Iteration:%d ******\n", pt);
    /* initialize the GPU */
        offset       = pt * pair_num[0] * seq_len; 
        score_offset = pt * pair_num[0] * 
            (seq_len+1) * (seq_len+1); 

        for (int i=0; i<config.num_streams; ++i) {
        /* Memcpy to device */
        
            se_time = gettime();
            cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set1[i],
            &sequence_set_rank[i][offset], sizeof(char)*pos1_rank[i][pair_num[i]],
            cudaMemcpyHostToDevice ) );
        
            cudaCheckError( __LINE__, cudaMemcpy( d_sequence_set2[i],
            &sequence_set_rank2[i][offset], sizeof(char)*pos2_rank[i][pair_num[i]],
            cudaMemcpyHostToDevice ) );
        
            cudaCheckError( __LINE__, cudaMemcpy( d_pos1[i], pos1_rank[i],
            sizeof(unsigned int)*(pair_num[i]+1), cudaMemcpyHostToDevice ) );
   
             cudaCheckError( __LINE__, cudaMemcpy( d_pos2[i], pos2_rank[i],
             sizeof(unsigned int)*(pair_num[i]+1), cudaMemcpyHostToDevice ) );
    
            cudaCheckError( __LINE__, cudaMemcpy( d_pos_matrix[i],
            pos_matrix_rank[i], sizeof(unsigned int)*(pair_num[i]+1),
            cudaMemcpyHostToDevice ) );
    
            cudaCheckError( __LINE__, cudaMemcpy( d_dim_matrix[i],                          dim_matrix[i],sizeof(unsigned int)*(pair_num[i]+1),                             cudaMemcpyHostToDevice ) );


            cudaCheckError( __LINE__, cudaStreamCreate( &(stream[i]) ));
        } // end for


            for (int i=0; i<config.num_streams; ++i ) {
                double stream_time_s, stream_time_e;
                stream_time_s = gettime();
                if (DEBUG) {
//                    fprintf(stderr,"Dataset[%d] starts\n", i);
                } // end if

                struct timeval s_tv, e_tv;
                gettimeofday(&s_tv, NULL);
                for (int j=0;j<1;j++)                
                nw_gpu(sequence_set_rank[i], sequence_set_rank2[i],
                pos1_rank[i], pos2_rank[i], score_matrix[i], pos_matrix_rank[i],                pair_num[i],d_score_matrix[i], stream[0], i, config.kernel);
                gettimeofday(&e_tv, NULL);
                double s_d = (double) s_tv.tv_sec + ((double) s_tv.tv_usec / 1000000.0);
                double e_d = (double) e_tv.tv_sec + ((double) e_tv.tv_usec / 1000000.0);

                printf("Kernel called: %.4f\n", e_d - s_d);
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
                    cudaCheckError( __LINE__, cudaStreamSynchronize(stream[0]));

                stream_time_e = gettime();
           } // end for
           ee_time  = gettime();
           pe_time += ee_time-se_time;

        printf("****** Iteration:%d end ******\n", pt);
    }
// Show time
        //shmem_barrier_all();
/*        fprintf(stderr,"Initialize GPU : %fs\n", p_time);
        fprintf(stderr,"Memory allocation and copy GPU : %fs\n", pd_time);*/
        char hostname[256];
        gethostname(hostname, 256);
        printf("PE[%d], host %s\n", Rank, hostname);
        printf("Time : %fs\n", pe_time);

// gather score matrix
/*    for ( int i=0; i<config.num_streams; ++i)
        sendtosingleprocess(full_score_matrix[i],                              score_num_pe, Rank, NumberOfProcessors);*/
    if (DEBUG) {
        for ( int i=0; i<config.num_streams; ++i) {
            int *score_matrix_cpu = (int *)malloc(
            pos_matrix_rank[i][pair_num[i]]*sizeof(int));
            
//computation on CPU
/*
            needleman_cpu(sequence_set_rank[i],
            sequence_set_rank2[i], pos1_rank[i], pos2_rank[i],                                score_matrix_cpu,pos_matrix_rank[i], pair_num[i],penalty);
                    if (validation(score_matrix_cpu, score_matrix[i],
                    pos_matrix[i][pair_num[i]]) )
                        printf("Stream %d - Validation: PASS\n", i);
                     else
                        printf("Stream %d - Validation: FAIL\n", i);
                    free(score_matrix_cpu);
*/
        }
    } //end DEBUG

   
        printf("\n\n");

        for (int i=0; i<config.num_streams; ++i ) {
           nw_gpu_destroy(i);
           cudaCheckError( __LINE__, cudaStreamDestroy ( stream[i] ));
        } // end for

        dataset=dataset-config.num_pairs[0];
        } // end while

        shmem_finalize();
        return 0;
    } // end main
