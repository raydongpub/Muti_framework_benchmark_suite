#include "matmul.h"
#include <unistd.h>
#include <iostream>
#include <sys/time.h>

using namespace std;
void display_mata(float *mat, int heg, int wid, int woid) {
    int total;
    cout << endl << "\t\tThe matrix display:" << endl;
    switch (woid) {
        case 0:
            total = heg * heg;
            for (int i=0; i<total; i++) {
                if (i%heg == 0) {
                    cout << "\t\t" << endl;
                }
                cout << mat[i] << " ";
            }
            cout << endl; break;
        case 1:
            total = heg * wid;
            for (int i=0; i<total; i++) {
                if (i%heg == 0) {
                    cout << "\t\t" << endl;
                }
                cout << mat[i] << " ";
            }
            cout << endl; break;
        case 2:
            total = heg * wid;
            for (int i=0; i<total; i++) {
                 if (i%wid == 0) {
                 cout << "\t\t" << endl;
            }
            cout << mat[i] << " ";
            }
            cout << endl; break;
    }

}

__global__ void ComputeMatrix_Kernel(int h, int w, int offset, int subnum, float *matA, float *matB, float *matC) {

    int row, col, idx;
    int tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    int gridsize   = gridDim.x * blockDim.x;
    float sum  = 0.0;
#if 0
    int stride = (subnum / gridsize) + 1;

    for (int i = 0; i < stride; i++) {
        idx = (i * gridsize) + tid;
#else
    for (int i = 0; i < subnum; i += gridsize) {
        idx = i + tid;
#endif
        row = idx / w;
        col = idx % w;
        if (idx < subnum) {
            for (int j = 0; j < h; j++) {
                sum += matA[row*h + j] * matB[h*col + j];
            }
            matC[idx] = sum; 
            sum = 0.0;
        }
    }
}

void CHK_ERR(int line, cudaError_t ce)
{
    if (ce != cudaSuccess){
        cout << "Error: line " << line << " "<< cudaGetErrorString(ce) << endl;    
    }
}

void cudaMatMul(int heg, int wid, int wh, int off, int subnum, float *matA, float *matB, float *matC, int peid_, int pid) {

    float *d_A, *d_B, *d_C;
    int size_A = wh * heg * sizeof(float);
    int size_B = wid * heg * sizeof(float);
    int s_size = subnum * sizeof(float); 
    cout << "\t\tTotal size: " << size_A + size_B + s_size << " wh: " << wh << " wid: " << wid<< " subnum: " << subnum <<endl;
    int peid   = pid;
   
    char hostname[128];
    gethostname(hostname, 128);
    int deviceCnt = 0;
    CHK_ERR( __LINE__, cudaGetDeviceCount(&deviceCnt));
    CHK_ERR( __LINE__, cudaSetDevice(peid % deviceCnt));
    cout << "\t\tPE:" << peid_ << "[" 
        << hostname << "]: RUN: Device[" << peid%deviceCnt << "]" << endl; 
     
    CHK_ERR( __LINE__, cudaMalloc(&d_A, size_A));
    CHK_ERR( __LINE__, cudaMalloc(&d_B, size_B));
    CHK_ERR( __LINE__, cudaMalloc(&d_C, s_size));

    CHK_ERR( __LINE__, cudaMemcpy(d_A, matA, size_A,
        cudaMemcpyHostToDevice));
    CHK_ERR( __LINE__, cudaMemcpy(d_B, matB, size_B,
        cudaMemcpyHostToDevice));
    CHK_ERR( __LINE__, cudaMemset(d_C, 0, s_size));

    cout << " Launching" << endl;
    struct timeval time_b, time_e;
    gettimeofday(&time_b, NULL);

    ComputeMatrix_Kernel <<<16, 128>>>(heg, wid, off, subnum, 
        d_A, d_B, d_C);
     
    CHK_ERR( __LINE__, cudaDeviceSynchronize());
    gettimeofday(&time_e, NULL);
    cout << "Kernel time: " << (time_e.tv_usec - time_b.tv_usec)*1e-6 + ((double)time_e.tv_sec - (double)time_b.tv_sec) << endl;

//    CHK_ERR( __LINE__, cudaMemcpy(matC, d_C, s_size, 
//        cudaMemcpyDeviceToHost));

    CHK_ERR( __LINE__, cudaFree(d_A));
    CHK_ERR( __LINE__, cudaFree(d_B));
    CHK_ERR( __LINE__, cudaFree(d_C));
//    display_mata(matC, wh, wid, 2);
    
}

