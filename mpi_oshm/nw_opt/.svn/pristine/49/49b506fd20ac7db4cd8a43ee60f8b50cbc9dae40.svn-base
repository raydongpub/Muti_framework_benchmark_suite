/*
* File:  global.c
 * Author: Da Li
 * Email:  da.li@mail.missouri.edu
 * Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file defines all the global variables.
*
*/
#include "global.h"

int DEBUG = 0;


struct _CONF_ config;
int * reference[MAX_STREAM][MAX_SEQ_NUM];
int * reference_cuda[MAX_SEQ_NUM];
int * matrix_cuda[MAX_SEQ_NUM];
int max_cols, max_rows;

char sequence_set1[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ] = {0};
char sequence_set1_cpu[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ] = {0};
char sequence_set2[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ] = {0};
char sequence_set2_cpu[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ] = {0};
unsigned int pos1[MAX_STREAM][MAX_SEQ_NUM] = {0};
unsigned int pos1_cpu[MAX_STREAM][MAX_SEQ_NUM] = {0};
unsigned int pos2[MAX_STREAM][MAX_SEQ_NUM] = {0};
unsigned int pos2_cpu[MAX_STREAM][MAX_SEQ_NUM] = {0};
unsigned int pos_matrix[MAX_STREAM][MAX_SEQ_NUM] = {0};
unsigned int pos_matrix_cpu[MAX_STREAM][MAX_SEQ_NUM] = {0};
unsigned int dim_matrix[MAX_STREAM][MAX_SEQ_NUM] = {0};
unsigned int dim_matrix_gpu[MAX_STREAM][MAX_SEQ_NUM] = {0};
unsigned int dim_matrix_cpu[MAX_STREAM][MAX_SEQ_NUM] = {0};

char * d_sequence_set1[MAX_STREAM];
char * d_sequence_set2[MAX_STREAM];
unsigned int * d_pos1[MAX_STREAM];
unsigned int * d_pos2[MAX_STREAM];
int * d_score_matrix[MAX_STREAM];
unsigned int * d_pos_matrix[MAX_STREAM];
unsigned int * d_dim_matrix[MAX_STREAM];
int pair_num_gpu[MAX_STREAM];
int pair_num_cpu[MAX_STREAM];
int penalty;
int maxLength;

double gettime() {
	struct timeval t;
	gettimeofday(&t,NULL);
	return t.tv_sec+t.tv_usec*1e-6;
}

