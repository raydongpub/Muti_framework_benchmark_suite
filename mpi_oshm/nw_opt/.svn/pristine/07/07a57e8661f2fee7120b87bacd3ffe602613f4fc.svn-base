/*
* File: needle_kernel_tile.cu 
 * Author: Da Li
 * Email: da.li@mail.missouri.edu
 * Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file defines kernel functions of tiled scan approach.
*
*/

#include <stdio.h>
#include <cuda.h>
#include "needle.h"

#if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

int tile_size = TILE_SIZE;

__device__ void print_s_tile( int s_tile[TILE_SIZE+1][TILE_SIZE+1], int row, int col)
{
	printf("%3d    %3d\n", s_tile[row-1][col-1], s_tile[row-1][col]);
	printf("%3d    %3d\n", s_tile[row][col-1], s_tile[row][col]);
	printf("\n\n");
}

__device__ void dump_s_tile( int s_tile[TILE_SIZE+1][TILE_SIZE+1])
{
	for (int i=0; i<=TILE_SIZE; ++i) {
		for (int j=0; j<=TILE_SIZE; ++j )
			printf("%3d\t", s_tile[i][j]);
		printf("\n");
	}
	printf("\n");

}
__global__ void needleman_cuda_init(int *score_matrix, unsigned int *pos_matrix, unsigned int *dim_matrix, int penalty)
{
	int pair_no = blockIdx.x;
	int tid = threadIdx.x;
	int *matrix = score_matrix+pos_matrix[pair_no];
	unsigned int row_size = dim_matrix[pair_no]+1;	// 1 element margin
	unsigned int stride = blockDim.x;

	int iteration = row_size / stride;
	// init first row
	for (int i=0; i<=iteration ; ++i)
	{	
		int index = (tid+stride*i);
		if ( index<row_size ){
			matrix[ index ] = index * penalty;
		}
	}
	// init first column
	for (int i=0; i<=iteration ; ++i)
	{
		int index = row_size * (tid+stride*i);
		if ( (tid+stride*i)<row_size) {
			matrix[ index ] = (tid+stride*i)*penalty;
		}
	}

}

__global__ void needleman_cuda_tile_upleft( char *sequence_set1, char *sequence_set2, 
									   unsigned int *pos1, unsigned int *pos2,
									   int *score_matrix, unsigned int *pos_matrix, unsigned int *dim_matrix,
									   int pair_num, int iter_no, int penalty)
{
	// 4KB, seq1[], sqe2[], tile[][]
	__shared__ char s_seq1[TILE_SIZE];
	__shared__ char s_seq2[TILE_SIZE];
	__shared__ int s_tile[(TILE_SIZE+1)][(TILE_SIZE+1)];

	int tile_num = iter_no;
	int total_tile = pair_num * tile_num;
	int tile_per_block = total_tile / gridDim.x + 1;
	for (int b = 0; b < tile_per_block; ++b ) {
		int tile_index = b * gridDim.x + blockIdx.x ;	
		int pair_no = tile_index / iter_no;
		int tile_no = tile_index % iter_no;
		if ( tile_index>=total_tile ) continue;
	int tid = threadIdx.x;

	char *seq1 = sequence_set1 + pos1[pair_no];
	char *seq2 = sequence_set2 + pos2[pair_no];
	int seq1_len = pos1[pair_no+1] - pos1[pair_no]; 
	int seq2_len = pos2[pair_no+1] - pos2[pair_no];
	int *matrix = score_matrix+pos_matrix[pair_no];
	unsigned int row_size = dim_matrix[pair_no] + 1;
	
	// calculate index, what are index_x & index_y
	int index_x = TILE_SIZE*tile_no + 1;	// 2-D matrix starts from (1,1)
	int index_y = TILE_SIZE*(iter_no-1) - TILE_SIZE*tile_no + 1;

	// load seq1
	int seq_index = index_x - 1 + tid; 	
	if ( tid<TILE_SIZE && seq_index<seq1_len )
		s_seq1[tid] = seq1[seq_index];
	// load seq2
	seq_index = index_y - 1 + tid;
	if ( tid<TILE_SIZE && seq_index<seq2_len )
		s_seq2[tid] = seq2[seq_index];

	// load boundary of tile
	if ( tid<TILE_SIZE ){
		int index = (index_y-1)*row_size + index_x + tid;	// x-index in 1-D array
		s_tile[0][tid+1] = matrix[index];
		//printf("s_tile[0][%d] = %d\n", tid+1, matrix[index]);

		index = (index_y+tid)*row_size + index_x - 1;		// y-index in 1-D array
		s_tile[tid+1][0] = matrix[index];
	}
	if ( tid==0 ) {
		int index = (index_y-1)*row_size + index_x-1;
		s_tile[tid][0] = matrix[index];
	}
	__syncthreads();
	// calculate
	for( int i = 0 ; i < TILE_SIZE ; i++){   
	  if ( tid <= i ){
		  index_x =  tid + 1;
		  index_y =  i - tid + 1;
          s_tile[index_y][index_x] = maximum(s_tile[index_y-1][index_x] + penalty,	//	up
											s_tile[index_y][index_x-1] + penalty,	//	left
						 					s_tile[index_y-1][index_x-1]+blosum62[s_seq2[index_y-1]][s_seq1[index_x-1]]);
	  }
	  __syncthreads(); 
    }
	for( int i = TILE_SIZE - 1 ; i >=0 ; i--){   
	  if ( tid <= i){
		  index_x =  tid + TILE_SIZE - i ;
		  index_y =  TILE_SIZE - tid;
          s_tile[index_y][index_x] = maximum(s_tile[index_y-1][index_x] + penalty,	//	up
											s_tile[index_y][index_x-1] + penalty,	//	left
						 					s_tile[index_y-1][index_x-1]+blosum62[s_seq2[index_y-1]][s_seq1[index_x-1]]);
	  }
	  __syncthreads();
	}

	int stride = blockDim.x / TILE_SIZE ;
	int row_iter = TILE_SIZE/stride+1;
	if ( tid < stride*TILE_SIZE ) {
		for ( int i=0; i<row_iter; ++i) {
			int s_tile_idx = tid % TILE_SIZE;
			int s_tile_idy = i * stride + tid / TILE_SIZE;
			if ( s_tile_idx<TILE_SIZE && s_tile_idy<TILE_SIZE) {
				index_x = TILE_SIZE*tile_no + 1 + s_tile_idx;// 2-D matrix starts from (1,1)
				index_y = TILE_SIZE*(iter_no-1) - TILE_SIZE*tile_no + 1 + s_tile_idy;
				matrix[index_x + index_y * row_size] = s_tile[s_tile_idy+1][s_tile_idx+1];
			}
		}
	}
	} // end for
}

__global__ void needleman_cuda_tile_bottomright(char *sequence_set1, char *sequence_set2, 
												unsigned int *pos1, unsigned int *pos2,
												int *score_matrix, unsigned int *pos_matrix, unsigned int *dim_matrix,
												int pair_num, int iter_no, int penalty)
{
	// 4KB, seq1[], sqe2[], tile[][]
	__shared__ char s_seq1[TILE_SIZE];
	__shared__ char s_seq2[TILE_SIZE];
	__shared__ int s_tile[(TILE_SIZE+1)][(TILE_SIZE+1)];

	int tile_num = iter_no;
	int total_tile = pair_num * tile_num;
	int tile_per_block = total_tile / gridDim.x + 1;
	for (int b = 0; b < tile_per_block; ++b ) {
        int tile_index = b * gridDim.x + blockIdx.x ;
        int pair_no = tile_index / iter_no;
        int tile_no = tile_index % iter_no;
        if ( tile_index>=total_tile ) continue;
    int tid = threadIdx.x;

	char *seq1 = sequence_set1 + pos1[pair_no];
	char *seq2 = sequence_set2 + pos2[pair_no];
	int seq1_len = pos1[pair_no+1] - pos1[pair_no]; 
	int seq2_len = pos2[pair_no+1] - pos2[pair_no];
	int *matrix = score_matrix+pos_matrix[pair_no];
	unsigned int row_size = dim_matrix[pair_no] + 1;
	// calculate index
	int index_x = row_size - TILE_SIZE*iter_no + TILE_SIZE*tile_no;	// 2-D matrix starts from (1,1)
	int index_y = row_size - TILE_SIZE - TILE_SIZE*tile_no;

	// load seq1
	int seq_index = index_x -1 + tid; 	
	if ( tid<TILE_SIZE && seq_index<seq1_len )
		s_seq1[tid] = seq1[seq_index];
	// load seq2
	seq_index = index_y -1 + tid;
	if ( tid<TILE_SIZE && seq_index<seq2_len )
		s_seq2[tid] = seq2[seq_index];

	// load boundary of tile
	if ( tid<TILE_SIZE ) {
		int index = (index_y-1)*row_size + index_x + tid;	// x-index in 1-D array
		s_tile[0][tid+1] = matrix[index];

		index = (index_y+tid)*row_size + index_x - 1;		// y-index in 1-D array
		s_tile[tid+1][0] = matrix[index];
	}
	if ( tid==0 ) {
		int index = (index_y-1)*row_size + index_x-1;
		s_tile[tid][0] = matrix[index];
	}
	__syncthreads();
	// calculate
	for( int i = 0 ; i < TILE_SIZE ; i++){
		if ( tid <= i ){
			index_x =  tid + 1;
			index_y =  i - tid + 1;
			s_tile[index_y][index_x] = maximum(s_tile[index_y-1][index_x] + penalty,//	up
											s_tile[index_y][index_x-1] + penalty,	//	left
						 					s_tile[index_y-1][index_x-1]+blosum62[s_seq2[index_y-1]][s_seq1[index_x-1]]);		}
		__syncthreads();  
	}
	for( int i = TILE_SIZE - 1 ; i >=0 ; i--){   
		if ( tid <= i){
			index_x =  tid + TILE_SIZE - i ;
			index_y =  TILE_SIZE - tid;
			s_tile[index_y][index_x] = maximum(s_tile[index_y-1][index_x] + penalty,//	up
											s_tile[index_y][index_x-1] + penalty,	//	left
											s_tile[index_y-1][index_x-1]+blosum62[s_seq2[index_y-1]][s_seq1[index_x-1]]);		}
		__syncthreads();
	}
	int stride = blockDim.x / TILE_SIZE ;
	int row_iter = TILE_SIZE/stride+1;
	if ( tid < stride*TILE_SIZE ) {
		for ( int i=0; i<row_iter; ++i) {
			int s_tile_idx = tid % TILE_SIZE;
			int s_tile_idy = i * stride + tid / TILE_SIZE;
			if ( s_tile_idx<TILE_SIZE && s_tile_idy<TILE_SIZE) {
				index_x = row_size - TILE_SIZE*iter_no + TILE_SIZE*tile_no + s_tile_idx;
				index_y = row_size - TILE_SIZE- TILE_SIZE*tile_no + s_tile_idy;
				matrix[index_x + index_y * row_size] = s_tile[s_tile_idy+1][s_tile_idx+1];
			}
		}
	}
	} // end for
}
