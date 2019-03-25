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
#if 0
char * sequence_set1[MAX_STREAM];
char * sequence_set2[MAX_STREAM];
unsigned int * pos1[MAX_STREAM];
unsigned int * pos2[MAX_STREAM];
unsigned int * pos_matrix[MAX_STREAM];
unsigned int dim_matrix[MAX_STREAM][MAX_SEQ_NUM] = {0};
#else
char sequence_set1[MAX_STREAM][MAX_SEQ_LEN * MAX_SEQ_NUM];
char sequence_set2[MAX_STREAM][MAX_SEQ_LEN * MAX_SEQ_NUM];
unsigned int pos1[MAX_STREAM][MAX_SEQ_NUM];
unsigned int pos2[MAX_STREAM][MAX_SEQ_NUM];
unsigned int pos_matrix[MAX_STREAM][MAX_SEQ_NUM];
unsigned int dim_matrix[MAX_STREAM][MAX_SEQ_NUM];
#endif

char * d_sequence_set1[MAX_STREAM];
char * d_sequence_set2[MAX_STREAM];
unsigned int * d_pos1[MAX_STREAM];
unsigned int * d_pos2[MAX_STREAM];
int * d_score_matrix[MAX_STREAM];
unsigned int * d_pos_matrix[MAX_STREAM];
unsigned int * d_dim_matrix[MAX_STREAM];
int pair_num[MAX_STREAM];
int penalty;
int maxLength;

double gettime() {
	struct timeval t;
	gettimeofday(&t,NULL);
	return t.tv_sec+t.tv_usec*1e-6;
}

