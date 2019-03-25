#include <stdio.h>
#include "needle.h"
#include <unistd.h>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>

using namespace std;

/******************** Variable *************************/
#define MAX_SEQ_LEN 3000
#define MAX_SEQ_NUM 7000
#define MAX_REPEAT 200

    #if defined (__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
    #endif





/******************* Util Func *************************/


inline void cudaCheckError(int line, cudaError_t ce){

    if (ce != cudaSuccess){
        printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
    } // end if
}  //end checkerror()


/******************** Kernel Func ***********************/

    __device__ void dia_upperleft(char *s_seq1, unsigned int seq1_len,
                                                                char *s_seq2, unsigned int seq2_len,
                                                                int *matrix, unsigned int dia_len,     
                                                                int *s_dia1, int *s_dia2, int *s_dia3,
                                                                int penalty)
    {
        int tid = threadIdx.x;
        int stripe = blockDim.x;
        int index_x;
        int index_y;
        int iteration;
        int *p_dia1, *p_dia2, *p_dia3, *p_tmp;
        // process the left-up triangle
        s_dia1[0] = matrix[0] = 0;
        s_dia2[0] = matrix[1] = penalty * 1;
        s_dia2[1] = matrix[1*(seq1_len+1)] = penalty * 1;

        p_dia1 = s_dia1;
        p_dia2 = s_dia2;
        p_dia3 = s_dia3;
        for (int i=2; i<=seq2_len; ++i){        // ith diagonal line            
                iteration = (i+1)/blockDim.x+1;
                if ( (i+1)%blockDim.x != 0 )    iteration++;
                for (int j=0; j<iteration; ++j) {
                        if ( tid+stripe*j<=i ) {        // ith diagonal has i+1 elements
                            for (int rr=0; rr<MAX_REPEAT; rr++) {
                                index_x = i-(tid+stripe*j);     index_y = tid+stripe*j;
                                if ( index_y==0 || index_y==i ) p_dia3[ index_y ] =  penalty * i;
                                else {
                                        p_dia3[ index_y ] =             \
                                                maximum(p_dia2[ index_y ] + penalty,    // up
                                                    p_dia2[ index_y-1 ] + penalty,      // left
                                                        p_dia1[ index_y-1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
                                }
                            }
                                // store to global memory
                                matrix[ index_x*(seq1_len+1)+index_y ] = p_dia3[ index_y ];
                        }
                }
                /*if ( threadIdx.x == 0 && blockIdx.x==0 ) {
                        for (int j=0; j<i+1; ++j)
                                printf("%d\t", p_dia3[j]);
                        printf("\n");
                }*/
                p_tmp = p_dia1;
                p_dia1 = p_dia2;
                p_dia2 = p_dia3;
                p_dia3 = p_tmp;
                __syncthreads();
        }
    }

 __device__ void dia_lowerright( char *s_seq1, unsigned int seq1_len,
                                                                char *s_seq2, unsigned int seq2_len,
                                                                int *matrix, unsigned int dia_len,     
                                                                int *s_dia1, int *s_dia2, int *s_dia3,
                                                                unsigned int start, int penalty)
    {
        int tid = threadIdx.x;
        int stripe = blockDim.x;
        int index_x, index_y;
        int iteration = dia_len/blockDim.x+1;
        int *p_dia1, *p_dia2, *p_dia3, *p_tmp;
        if ( dia_len%blockDim.x!=0 ) iteration++;
        // initial, load from shared memory
        for (int i=0; i<iteration; ++i) {
                if ( tid+stripe*i<seq1_len+1 ) {
                        index_x = seq2_len - (tid+stripe*i);    index_y = (tid+stripe*i);
                        s_dia1[ tid+stripe*i ] = matrix[ index_x*(seq1_len+1)+index_y ];
                }
        }
        /*if ( threadIdx.x == 0 && blockIdx.x==0 ) {
                for (int j=0; j<seq1_len+1; ++j)
                        printf("%d\t", s_dia1[j]);
                printf("\n");
        }*/
        __syncthreads();
        p_dia1 = s_dia1;
        p_dia2 = s_dia2;
        p_dia3 = s_dia3;
        // calculate the 1th diagonal
        for (int i=0; i<iteration; ++i) {
                if ( tid+stripe*i<seq1_len ) {
                    for (int rr=0; rr<MAX_REPEAT; rr++) {
                        index_x = seq2_len - (tid+stripe*i);    index_y = 1 + (tid+stripe*i);
                        p_dia2[ tid+stripe*i ] = \
                                                maximum(p_dia1[ tid+stripe*i+1 ] + penalty,     // up
                                                            p_dia1[ tid+stripe*i ] + penalty,   // left
                                                                matrix[(index_x-1)*(seq1_len+1)+index_y-1]+blosum62[s_seq2[index_x]][s_seq1[index_y]] );
                    }
                        matrix[ index_x*(seq1_len+1)+index_y ] = p_dia2[ tid+stripe*i ];
                }
        }
        __syncthreads();
        for (int i=2; i<=seq1_len; ++i){// ith diagonal line, start from 2
                iteration = (seq1_len-i+1)/blockDim.x;
                if ( (seq1_len-i+1)%blockDim.x != 0 )   iteration++;
                for (int j=0; j<iteration; ++j) {
                        index_x = seq2_len - (tid+stripe*j);
                        index_y =  i + (tid+stripe*j);
                        if ( tid+stripe*j +i <seq1_len+1 ) {
                            for (int rr=0; rr<MAX_REPEAT; rr++) {
                                p_dia3[ tid+stripe*j ] =        \
                                        maximum(p_dia2[ tid+stripe*j+1 ] + penalty,     // up
                                                    p_dia2[ tid+stripe*j ] + penalty,   // left
                                                        p_dia1[ tid+stripe*j+1 ]+blosum62[ s_seq2[index_x] ][ s_seq1[index_y] ] );
                            }
                                // store to global memory
                                matrix[ index_x*(seq1_len+1)+index_y ] = p_dia3[ tid+stripe*j ];
                        }
                }
                p_tmp = p_dia1;
                p_dia1 = p_dia2;
                p_dia2 = p_dia3;
                p_dia3 = p_tmp;
                __syncthreads();
        }
    }


    __global__ void needleman_cuda_diagonal(char *sequence_set1, char *sequence_set2,
                                                                           unsigned int *pos1, unsigned int *pos2,
                                                                           int *score_matrix, unsigned int *pos_matrix,
                                                                           unsigned int max_pair_no, int penalty)
    {
        int pair_no, seq1_len, seq2_len;
        int tid = threadIdx.x;
        // 48 KB/4 = 12KB, seq1+sqe2, diagonal1, diagonal2, diagonal3
        __shared__ char s_seq1[MAX_SEQ_LEN];
        __shared__ char s_seq2[MAX_SEQ_LEN];
        __shared__ int s_dia1[MAX_SEQ_LEN];
        __shared__ int s_dia2[MAX_SEQ_LEN];
        __shared__ int s_dia3[MAX_SEQ_LEN];
        char *seq1;
        char *seq2;
        int *matrix;

        int pair_per_block = max_pair_no / gridDim.x + 1;

        for (int k=0; k<pair_per_block; ++k) {
                pair_no = k * gridDim.x + blockIdx.x;   // for each block, caculate one pair
                if ( pair_no<max_pair_no ) {
                        seq1 = sequence_set1 + pos1[pair_no];
                        seq2 = sequence_set2 + pos2[pair_no];
                        matrix = score_matrix+pos_matrix[pair_no];
                        seq1_len = pos1[pair_no+1] - pos1[pair_no];
                        seq2_len = pos2[pair_no+1] - pos2[pair_no];

                        // load the two sequences
                        unsigned int stride_length = blockDim.x;
                        for (int i=0; i<seq1_len/stride_length+1; ++i){
                                if ( tid+i*stride_length<seq1_len )
                                        s_seq1[tid+i*stride_length+1] = seq1[tid+i*stride_length];
                        }
                        for (int i=0; i<seq2_len/stride_length+1; ++i){
                                if ( tid+i*stride_length<seq2_len )
                                s_seq2[tid+i*stride_length+1] = seq2[tid+i*stride_length];
                        }
                }
                __syncthreads();

                if ( pair_no<max_pair_no ) {
                        dia_upperleft( s_seq1, seq1_len, s_seq2, seq2_len, matrix, seq2_len,
                                                        s_dia1, s_dia2, s_dia3, penalty);

                        dia_lowerright( s_seq1, seq1_len, s_seq2, seq2_len, matrix, seq1_len+1,
                                                        s_dia1, s_dia2, s_dia3, 1, penalty);
                }
        }
    }


/******************** wrapper Func *********************/
void nw_gpu(char * sequence_set1, char * sequence_set2, unsigned int * pos1, unsigned int * pos2,
                        unsigned int * pos_matrix, unsigned int pair_num,
                        int * score_matrix,
                        int num_threads, int num_blocks, int penalty, cudaStream_t stream) {
        needleman_cuda_diagonal<<<pair_num, num_threads, 0, stream>>>(
            sequence_set1, sequence_set2, pos1, pos2,
            score_matrix, pos_matrix, pair_num, penalty);

        cudaCheckError( __LINE__, cudaDeviceSynchronize());
        
}

void cudaMatMul(int num_threads, int num_blocks, int penalty, int pair_num, int offset, char * seq_data1, char * seq_data2, unsigned int * pos1, unsigned int * pos2, unsigned int * pos_matrix, int dpeid, int pid) {

    // cuda Variable

    char * d_sequence_set1;
    char * d_sequence_set2;
    unsigned int * d_pos1;
    unsigned int * d_pos2;
    int * d_score_matrix;
    unsigned int * d_pos_matrix;
   
    cout << "cuda0" << endl;
    // cudaSetDevice
    char hostname[128];
    gethostname(hostname, 128);
    int deviceCnt = 0;
    cudaCheckError( __LINE__, cudaGetDeviceCount(&deviceCnt));
    
//    cudaCheckError( __LINE__, cudaSetDevice(0));
    cudaCheckError( __LINE__, cudaSetDevice(pid % deviceCnt));
    cout << "PE:" << pid << "["
        << hostname << "]: RUN: Device[" << pid%deviceCnt << "]"  << "penalty: " << penalty << " "<< num_blocks << ":" << num_threads << endl;
    
    // cudaMalloc for data
    cudaCheckError( __LINE__,
        cudaMalloc( (void**)&d_sequence_set1,
        sizeof(char)*pos1[pair_num] ) );
    printf("SIZE 0: %d\n", sizeof(char)*pos1[pair_num]);


    cudaCheckError( __LINE__,
        cudaMalloc( (void**)&d_sequence_set2,
        sizeof(char)*pos2[pair_num] ) );
    printf("SIZE 1: %d\n", sizeof(char)*pos2[pair_num]);

    cudaCheckError( __LINE__,
        cudaMalloc( (void**)&d_score_matrix,
        sizeof(int)*pos_matrix[pair_num]) );
    printf("SIZE 2: %d\n", sizeof(int)*pos_matrix[pair_num]);

    cudaCheckError( __LINE__,
        cudaMalloc( (void**)&d_pos1,
        sizeof(unsigned int)*(pair_num+1) ) );
    printf("SIZE 3: %d\n", sizeof(unsigned int)*(pair_num+1));


    cudaCheckError( __LINE__,
        cudaMalloc( (void**)&d_pos2,
        sizeof(unsigned int)*(pair_num+1) ) );
    printf("SIZE 4: %d\n", sizeof(unsigned int)*(pair_num+1));


    cudaCheckError( __LINE__,
        cudaMalloc( (void**)&d_pos_matrix,
        sizeof(unsigned int)*(pair_num+1) ) );
    printf("SIZE 5: %d\n", sizeof(unsigned int)*(pair_num+1));


    // cudaMemcpy for data
    cudaCheckError( __LINE__,
        cudaMemcpy( d_sequence_set1,
        &seq_data1[offset],
        sizeof(char)*pos1[pair_num],
        cudaMemcpyHostToDevice ) );

    cudaCheckError( __LINE__,
        cudaMemcpy( d_sequence_set2,
        &seq_data2[offset],
        sizeof(char)*pos2[pair_num],
        cudaMemcpyHostToDevice ) );

    cudaCheckError( __LINE__,
        cudaMemcpy( d_pos1, pos1,
        sizeof(unsigned int)*(pair_num+1),
        cudaMemcpyHostToDevice ) );

    cudaCheckError( __LINE__,
        cudaMemcpy( d_pos2, pos2,
        sizeof(unsigned int)*(pair_num+1),
        cudaMemcpyHostToDevice ) );

    cudaCheckError( __LINE__,
        cudaMemcpy( d_pos_matrix,
        pos_matrix,
        sizeof(unsigned int)*(pair_num+1),
        cudaMemcpyHostToDevice ) );

    // cudaStream
    cudaStream_t stream;
    cudaCheckError( __LINE__,
        cudaStreamCreate( &stream ));

    // kernel

    struct timeval s_t, e_t;

    gettimeofday(&s_t, NULL);
    for (int j=0; j<1; j++) 
        nw_gpu(d_sequence_set1,
            d_sequence_set2,
            d_pos1, d_pos2,
            d_pos_matrix,
            pair_num,d_score_matrix,
            num_threads, num_blocks, penalty, stream);

    gettimeofday(&e_t, NULL);
    double s_time, e_time;
    s_time = (double)s_t.tv_sec + (double)s_t.tv_usec/1000000.0;
    e_time = (double)e_t.tv_sec + (double)e_t.tv_usec/1000000.0;

    cout << "\t\t\t----kernel: " << e_time - s_time << endl;
    cudaCheckError( __LINE__,
        cudaStreamSynchronize(stream));
    cudaCheckError( __LINE__,
        cudaStreamDestroy(stream));
    // cudaFree
    cudaCheckError( __LINE__,
        cudaFree(d_sequence_set1));
    cudaCheckError( __LINE__,
        cudaFree(d_sequence_set2));
    cudaCheckError( __LINE__,
        cudaFree(d_score_matrix));
    cudaCheckError( __LINE__,
        cudaFree(d_pos1));
    cudaCheckError( __LINE__,
        cudaFree(d_pos2));
    cudaCheckError( __LINE__,
        cudaFree(d_pos_matrix));

}
