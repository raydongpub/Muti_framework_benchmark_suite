/*
* File:  nw_gpu.h
* Author: Da Li
* Email:  da.li@mail.missouri.edu
* Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file has all the declarations of GPU related functions.
*
*/

#ifndef __NW_GPU_H__
#define __NW_GPU_H__

#include "global.h"

void nw_cuda_diagonal(cudaStream_t stream, int stream_num);

void nw_cuda_tile(cudaStream_t stream, int stream_num);

void nw_gpu_allocate(int Rank);

void nw_gpu_copyback(int *score_matrix, int *d_score_matrix, unsigned int *pos_matrix, unsigned int pair_num, int stream_num);

void nw_gpu_destroy(int stream_num);

void nw_gpu(char * sequence_set1, char * sequence_set2, unsigned int * pos1, unsigned int * pos2, 
            int * score_matrix, unsigned int * pos_matrix, unsigned int pair_num,
            int * d_score_matrix, cudaStream_t stream, int stream_num, 
			int kernel_type);
#endif

