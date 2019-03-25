#include "cuda_runtime_api.h"
#include "main.h"
#include "mmMpicudakernel.cuh"
#include "matdataset.h"
#include "mpi.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define MPI_NN_ROOT 0

#define MPI_NN_ROOT 0
#ifndef _DOUBLE_PRECISION
#define MPI_PRECISION MPI_FLOAT
#else //_DOUBLE_PRECISION
#define MPI_PRECISION MPI_DOUBLE
#endif //_DOUBLE_PRECISION

using namespace std;

MatrixDataset * plMatrix;
MatrixDataset * rlMatrix;
pConfig       * mpConfig;

int rank, commSize;
int Numelements, numelements, wid, heg;

int LibSetup(MatrixDataset * pMatrix, pConfig * pConf) {
    mpConfig = pConf;
    rank     = mpConfig->mrank;
    commSize = mpConfig->mcommSize;
    if (mpConfig->mrank == 0)
        plMatrix = pMatrix;
    return 0;
}

PRECISION * CollectiveInitialize(parConfig * par_mat) {
/**
* Pass the variale
**/
    PRECISION * vec_b;
    if (rank == MPI_NN_ROOT) {
        Numelements = plMatrix->mNumelements;
        wid         = plMatrix->mpMatrix.width;
        heg         = plMatrix->mpMatrix.height;
        vec_b       = plMatrix->mpMatrix.elements+Numelements/2;
    }
    MPI_Bcast(&Numelements, 1, MPI_INT, MPI_NN_ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&wid, 1, MPI_INT, MPI_NN_ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&heg, 1, MPI_INT, MPI_NN_ROOT, MPI_COMM_WORLD);
    numelements = Numelements / 2;  
/**
 * Get Partition factor
 */
   int div_fac           = wid / commSize;
   int mat_fraction      = wid * div_fac;
   par_mat->div_fac      = div_fac;
   par_mat->mat_fraction = mat_fraction;
   par_mat->wid          = wid;
/**
* Allocate shared space for all PE
**/

    int oshm = false;
    rlMatrix = new MatrixDataset(numelements, wid, heg, oshm);
/*    if (rank) {
        plMatrix = new MatrixDataset(Numelements, wid, heg, oshm);
    }*/
// Create localbufer for part of vec_b
    PRECISION * localB = new PRECISION[mat_fraction];
// Distribute vec_b
    if (rank==0) {
        memcpy(localB, vec_b, mat_fraction*sizeof(PRECISION));
        for (int i=1; i<commSize; i++)
            MPI_Send(&vec_b[i*mat_fraction], mat_fraction, 
                MPI_PRECISION, i, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Recv(localB, mat_fraction, MPI_PRECISION, 0, 0, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return localB; 
}

int LibEntry(bool showmat, int num_blocks, int num_threads, MatrixDataset * pMatrix, pConfig * pConf) {
    
    int col,row;
    bool oshm = true, result = true;
    int mstart, mnum;
    PRECISION * local_a_buf, * local_b_buf, * local_c_buf, * vec_a;

    parConfig * par_mat = new parConfig[1];
    local_b_buf = CollectiveInitialize(par_mat);    
/**
* cuda error check
**/
#define CHK_ERR(str)                                             \
    do {                                                         \
        cudaError_t ce = str;                                    \
        if (ce != cudaSuccess) {                                 \
            cout << "Error: " << cudaGetErrorString(ce) << endl; \
            exit(-1);                                            \
        }                                                        \
    } while (0)
// Partition cell info
   cout << "**** Cell info ****" << endl;
   cout << "cell size: " << par_mat->div_fac 
        << "\tvec size: " << par_mat->mat_fraction << endl << endl;


    int cell_size = par_mat->div_fac*(par_mat->div_fac); 
// Create localbufer for part of vec_a
    local_a_buf = new PRECISION[par_mat->mat_fraction];
// Create vec_c data structurue
    local_c_buf = new PRECISION[cell_size];
    if (rank == MPI_NN_ROOT) 
        vec_a = plMatrix->mpMatrix.elements;
    for (int i=0; i<commSize; i++) {
// Distribute vec_a
        if (rank == MPI_NN_ROOT) {
            memcpy(local_a_buf, (vec_a+i*(par_mat->mat_fraction)),
                (par_mat->mat_fraction*sizeof(PRECISION)));
        }
        MPI_Bcast(local_a_buf, par_mat->mat_fraction, 
            MPI_PRECISION, 0, MPI_COMM_WORLD);
/**
* Initialize the cuda part code
**/
        printf("\trank = %d, iter = %d, begin CUDA init...\n", rank, commSize);
        CHK_ERR(    CudaInitialize(local_a_buf, local_b_buf, 
            local_c_buf, par_mat, rank, commSize));
        CHK_ERR(    ComputeMatrixAttributes(num_blocks, num_threads));
        printf("\trank = %d, iter = %d, complete CUDA kernel...\n", rank, commSize);
        MPI_Barrier(MPI_COMM_WORLD);
        CHK_ERR(    CudaClean());
/*        if (!peid && showmat)
            rlMatrix->showmatrix(result);*/
        printf("\trank = %d, finish CUDA round...\n", rank);
/**
 * Merge the small vec_c into the result matrix
 */
/**** NOTE: need to merge result ****/
/*        for (int ib=0; ib<cell_size; ib++ ) {

        }
*/

    }
}

int LibCleanUp(void) {
    if (!rank) {
        rlMatrix->SaveToFile("mpi_cuda.bin");
        cout << "CleanUp!" << endl;\
        delete plMatrix;
    }
    delete rlMatrix;

}

