/*
* File:  nw-multi-stream.cu
* Author: Da Li
* Email:  da.li@mail.missouri.edu
* Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
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

//#define TRACEBACK

extern char blosum62_cpu[24][24];
extern int tile_size;

void init_conf() {
	config.debug = false;
	config.device = 0;
	config.kernel = 0;
	config.cpu_threads = 1;
	config.fraction = 20;
	config.num_streams = 0;
	config.num_blocks = 28;
	config.num_threads = 32;
	config.length = 1600;
	config.penalty = -10;
	config.repeat = 3;
}

void init_device(int device) {
	cudaSetDevice(device);
	cudaFree(0);
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "\nUsage: %s [options]\n", argv[0]);
    fprintf(stderr, "\t[--length|-l <length>] - x and y length (default: %d)\n",config.length);
    fprintf(stderr, "\t[--penalty|-p <penalty>] - penalty (negative integer, default: %d)\n",config.penalty);
    fprintf(stderr, "\t[--num_pairs|-n <number>] - number of pairs per stream\n");
    fprintf(stderr, "\t[--cpu_threads|-T <threads>]- threads number on CPU (default: %d)\n",config.cpu_threads);
	fprintf(stderr, "\t[--f_cpu|-f <number>]- fraction of pairs on CPU (default: %d)\n",config.fraction);
	fprintf(stderr, "\t[--device|-d <device>]- device ID (default: %d)\n",config.device);
    fprintf(stderr, "\t[--kernel|-k <kernel type>]- 0: diagonal 1: tile (default: %d)\n",config.kernel);
    fprintf(stderr, "\t[--num_blocks|-b <blocks>]- blocks number per grid (default: %d)\n",config.num_blocks);
    fprintf(stderr, "\t[--num_threads|-t <threads>]- threads number per block (default: %d)\n",config.num_threads);
    fprintf(stderr, "\t[--repeat|-r <num>]- repeating number (default: %d)\n",config.repeat);
    fprintf(stderr, "\t[--debug]- 0: no validation 1: validation (default: %d)\n",config.debug);
    fprintf(stderr, "\t[--help|-h]- Help information \n");
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
    	fprintf(stderr, "Case %d - sequence number = %d\n", i, config.num_pairs[i]);
	}
	fprintf(stderr, "threads CPU = %d\n", config.cpu_threads);
	fprintf(stderr, "fraction on CPU = %d\n", config.fraction);
	fprintf(stderr, "sequence length = %d\n", config.length);
    fprintf(stderr, "penalty = %d\n", config.penalty);
    fprintf(stderr, "block number = %d\n", config.num_blocks);
    fprintf(stderr, "thread number = %d\n", config.num_threads);
    fprintf(stderr, "repeat = %d\n", config.repeat);
    fprintf(stderr, "debug = %d\n", config.debug);
	printf("==============================================\n");
}

void validate_config()
{
    if ( config.length % tile_size != 0 &&
         config.kernel == 1 ) {
        fprintf(stderr, "Tile kernel used.\nSequence length should be multiple times of tile size %d\n", tile_size);
        exit(1);
    }
}

int validation(int *score_matrix_cpu, int *score_matrix, unsigned int length)
{
    unsigned int i = 0;
    while (i!=length){
        if ( score_matrix_cpu[i]==score_matrix[i] ){
            ++i;
            continue;
        }
        else {
            printf("On GPU: score_matrix[%d] = %d\n", i, score_matrix[i]);
            printf("On CPU: score_matrix[%d] = %d\n", i, score_matrix_cpu[i]);
            return 0;
        }
    }
    return 1;
}

int parse_arguments(int argc, char **argv)
{
	int i = 1;
	if (argc<2) {
		usage(argc, argv);
		return 0;
	}
	while(i<argc) {
		if(strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"device number missing.\n");
				return 0 ;
			}
			config.device = atoi(argv[i]);
		}else if(strcmp(argv[i], "--debug") == 0){
			config.debug = 1;
		}else if(strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--kernel") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"device number missing.\n");
				return 0 ;
			}
			config.kernel = atoi(argv[i]);
		}else if(strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--repeat") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"repeating number missing.\n");
				return 0 ;
			}
			config.repeat = atoi(argv[i]);
		}else if(strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--num_threads") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"thread number missing.\n");
				return 0 ;
			}
			config.num_threads = atoi(argv[i]);
		}else if(strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--num_blocks") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"block number missing.\n");
				return 0 ;
			}
			config.num_blocks = atoi(argv[i]);
		}else if(strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--penalty") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"penalty score missing.\n");
				return 0 ;
			}
			config.penalty = atoi(argv[i]);
		}else if(strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--num_pairs") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"number of paris missing.\n");
				return 0 ;
			}
			config.num_pairs[ config.num_streams ] = atoi(argv[i]);
			if ( config.num_pairs[ config.num_streams ] > MAX_SEQ_NUM ) {
				fprintf(stderr, "The maximum sequence number per stream is %d\n", MAX_SEQ_NUM);
				return 0;
			}
			config.num_streams++;
		}else if(strcmp(argv[i], "-f") ==0 || strcmp(argv[i], "--f_cpu") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"fraction on CPU missing.\n");
				return 0 ;
			}
			config.fraction = atoi(argv[i]);
			if ( config.fraction<0 || config.fraction>100 ) {
				fprintf(stderr,"Illegal fraction on CPU : %d%.\n", config.fraction);
				return 0;
			}
		}else if(strcmp(argv[i], "-T") == 0 || strcmp(argv[i], "--cpu_threads") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"threads number on cpu missing.\n");
				return 0 ;
			}
			config.cpu_threads = atoi(argv[i]);
		}else if(strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--lengths") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"sequence length missing.\n");
				return 0 ;
			}
			config.length = atoi(argv[i]);
			if ( config.length > MAX_SEQ_LEN ) {
				fprintf(stderr,"The maximum seqence length is %d\n", MAX_SEQ_LEN);
				return 0;
			}
		}else if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			usage(argc, argv);
			return 0;
		}
		else {
			fprintf(stderr, "Unrecognized option : %s\nTry --help for more information", argv[i]);
			return 0;
		}
		i++;
	}
	return 1;
}

int main(int argc, char **argv)
{
	double s_time, e_time;
	init_conf();
	while(!parse_arguments(argc, argv)) usage(argc, argv);

	print_config();
	validate_config();
	int dev_num = config.device;
	int	penalty = config.penalty;
	int seq1_len, seq2_len;
	int seq_len = config.length;
	DEBUG = config.debug;

	cudaStream_t stream[config.num_streams];

	int *score_matrix[config.num_streams];
	int *score_matrix_cpu[config.num_streams];
	srand ( 7 );
	/* generate data */
	for (int k=0; k<config.num_streams;++k) {
		pair_num_cpu[k] = config.num_pairs[k] * config.fraction / 100;
		pair_num_gpu[k] = config.num_pairs[k] - pair_num_cpu[k];
		fprintf(stderr, "stream %d:\n", k);
		fprintf(stderr, "    - On GPU : %d\n", pair_num_gpu[k]);
		fprintf(stderr, "    - On CPU : %d\n", pair_num_cpu[k]);
		/* On GPU */
    	pos_matrix[k][0] = pos1[k][0] = pos2[k][0] = 0;
   		for (int i=0; i<pair_num_gpu[k]; ++i){
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
        	//printf("Matrix size increase: %d\n", (seq1_len+1) * (seq2_len+1));
        	pos_matrix[k][i+1] = pos_matrix[k][i] + (seq1_len+1) * (seq2_len+1);
			dim_matrix[k][i] = (unsigned int)ceil( (float)seq_len/TILE_SIZE )*TILE_SIZE;
    	}
		score_matrix[k] = (int *)malloc( pos_matrix[k][pair_num_gpu[k]]*sizeof(int) );
	
		/* On CPU */
    	pos_matrix_cpu[k][0] = pos1_cpu[k][0] = pos2_cpu[k][0] = 0;
   		for (int i=0; i<pair_num_cpu[k]; ++i){
       		//please define your own sequence 1
       		seq1_len = seq_len; //64+rand() % 20;
        	//printf("Seq1 length: %d\n", seq1_len);    
        	for (int j=0; j<seq1_len; ++j)
            	sequence_set1_cpu[k][ pos1_cpu[k][i] + j ] = rand() % 20 + 1;
        	pos1_cpu[k][i+1] = pos1_cpu[k][i] + seq1_len;
        	//please define your own sequence 2.
        	seq2_len = seq_len;//64+rand() % 20;       
        	//printf("Seq2 length: %d\n\n", seq2_len);      
        	for (int j=0; j<seq2_len; ++j)
            	sequence_set2_cpu[k][ pos2_cpu[k][i] +j ] = rand() % 20 + 1;
        	pos2_cpu[k][i+1] = pos2_cpu[k][i] + seq2_len;
        	//printf("Matrix size increase: %d\n", (seq1_len+1) * (seq2_len+1));
        	pos_matrix_cpu[k][i+1] = pos_matrix_cpu[k][i] + (seq1_len+1) * (seq2_len+1);
			dim_matrix_cpu[k][i] = (unsigned int)ceil( (float)seq_len/TILE_SIZE )*TILE_SIZE;
    	}
		score_matrix_cpu[k] = (int *)malloc( pos_matrix_cpu[k][pair_num_cpu[k]]*sizeof(int) );
	}
	/* initialize the GPU */
	s_time = gettime();
	init_device( dev_num );
	e_time = gettime();
	fprintf(stderr, "Initialize GPU : %fs\n", e_time - s_time);

	s_time = gettime();
	for (int i=0; i<config.num_streams; ++i) {
		nw_gpu_allocate(i);
		cudaStreamCreate( &(stream[i]) );
	}
	e_time = gettime();
	fprintf(stderr, "Memory allocation and copy on GPU : %fs\n", e_time - s_time);
	
	for (int r=0; r<config.repeat; ++r) {
	fprintf(stderr, "Round #%d:\n", r);
	s_time = gettime();
	#pragma omp parallel for
	for (int i=0; i<config.num_streams; ++i ) {
		if ( pair_num_gpu[i]!=0 ) {
			nw_gpu(sequence_set1[i], sequence_set2[i], pos1[i], pos2[i], 
				   score_matrix[i], pos_matrix[i], pair_num_gpu[i], 
				   d_score_matrix[i], stream[i], i, config.kernel);
		}
	}
	for (int i=0; i<config.num_streams; ++i ) {
		if ( pair_num_cpu[i]!=0 ) {
    		needleman_cpu_omp(sequence_set1_cpu[i], sequence_set2_cpu[i], pos1_cpu[i], pos2_cpu[i], score_matrix_cpu[i], pos_matrix_cpu[i], pair_num_cpu[i], penalty);
		}
	}
	#pragma omp parallel for
	for (int i=0; i<config.num_streams; ++i ) {
		if ( pair_num_gpu[i]!=0 ) {
			nw_gpu_copyback(score_matrix[i], d_score_matrix[i], pos_matrix[i], pair_num_gpu[i], i);
		}
	}
	e_time = gettime();
	fprintf(stderr, "Computation on CPU & GPU : %fs\n", e_time - s_time);
	
	if (DEBUG) {
		for ( int i=0; i<config.num_streams; ++i) {
    		int *score_matrix_valid = (int *)malloc( pos_matrix[i][pair_num_gpu[i]]*sizeof(int));
    		needleman_cpu_omp(sequence_set1[i], sequence_set2[i], pos1[i], pos2[i], score_matrix_valid, pos_matrix[i], pair_num_gpu[i], penalty);
			if ( validation(score_matrix_valid, score_matrix[i], pos_matrix[i][pair_num_gpu[i]]) )
        		printf("Stream %d - On GPU Validation: PASS\n", i);
    		else
        		printf("Stream %d - On GPU Validation: FAIL\n", i);
			free(score_matrix_valid);
    		score_matrix_valid = (int *)malloc( pos_matrix_cpu[i][pair_num_cpu[i]]*sizeof(int));
    		needleman_cpu_omp(sequence_set1_cpu[i], sequence_set2_cpu[i], pos1_cpu[i], pos2_cpu[i], score_matrix_valid, pos_matrix_cpu[i], pair_num_cpu[i], penalty);
			if ( validation(score_matrix_valid, score_matrix_cpu[i], pos_matrix_cpu[i][pair_num_cpu[i]]) )
        		printf("Stream %d - On CPU Validation: PASS\n", i);
    		else
        		printf("Stream %d - On CPU Validation: FAIL\n", i);
			free(score_matrix_valid);
		}
	}
	}	// end for repeat
	printf("\n");
	for (int i=0; i<config.num_streams; ++i ) {
		nw_gpu_destroy(i);
		cudaStreamDestroy ( stream[i] );
	}

}
