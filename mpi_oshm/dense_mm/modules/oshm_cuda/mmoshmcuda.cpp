#include "cuda_runtime_api.h"
#include "main.h"
#include "mmoshmcudakernel.cuh"
#include "matdataset.h"
#include "mpi.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <shmem.h>
#include <string.h>
#include <stdlib.h>
using namespace std;

MatrixDataset * plMatrix;
MatrixDataset * rlMatrix;
MatrixDataset * shMatrix;

int numpes, peid;
int *Numelements, numelements, *wid, *heg;

int LibSetup(MatrixDataset * pMatrix, pConfig * pConf) {
    numpes   = pConf->mcommSize;
    peid     = pConf->mrank;
    return 0;
}


float * CollectiveInitialize(parConfig * par_mat, MatrixDataset * pMatrix, pConfig * pConf) {

    Numelements = (int *) shmem_malloc(sizeof(int));
    wid         = (int *) shmem_malloc(sizeof(int));
    heg         = (int *) shmem_malloc(sizeof(int));
#if 1
    if (peid == 0){
        *Numelements = pMatrix->mNumelements;
        *wid         = pMatrix->mpMatrix.width;
        *heg         = pMatrix->mpMatrix.height;
    }
#endif
/**
* Pass the variale
**/
    shmem_barrier_all();
    if (peid) {
        shmem_int_get(Numelements, Numelements, 1, 0);
        shmem_int_get(wid, wid, 1, 0);
        shmem_int_get(heg, heg, 1, 0);
    }

    numelements = *Numelements / 2;
/**
 *  * Get Partition factor
 *   */
   int div_fac           = *wid / numpes;
   int mat_fraction      = *wid * div_fac;
   par_mat->div_fac      = div_fac;
   par_mat->mat_fraction = mat_fraction;
   par_mat->wid          = *wid;
/**
* Allocate shared space for all PE
**/
    int oshm = true;
    //shMatrix = new MatrixDataset(*Numelements, *wid, *heg, oshm);
    //rlMatrix = new MatrixDataset(numelements, *wid, *heg, oshm);
// Create localbufer for part of vec_b
    
    //float * localB = (float *)shmem_malloc(mat_fraction*sizeof(float));
    float * localB, * tempB;
    localB = (float *)shmem_malloc(mat_fraction*sizeof(float));
    //tempB = (float *)shmem_malloc(numelements*sizeof(float));
    if (peid == 0) {
        memcpy(localB, &((pMatrix->mpMatrix.elements)[numelements]), mat_fraction * sizeof(float));
        //memcpy(tempB, &((pMatrix->mpMatrix.elements)[numelements]), numelements * sizeof(float));
    }
    shmem_barrier_all();    
    if (peid != 0) {
        shmem_getmem(localB, localB, mat_fraction*sizeof(float), 0/*(peid+1)%numpes*/);
    }
    shmem_barrier_all();    
    return localB;

}

int LibEntry(bool showmat, int num_blocks, int num_threads, MatrixDataset * pMatrix, pConfig * pConf) {
    
    int col,row;
    bool oshm = true, result = true;
    int mstart, mnum;
    float * local_a_buf, * local_b_buf, * local_c_buf, * vec_a, * tempbuf;
    long * pSync;

    pSync = (long *) shmem_malloc(_SHMEM_BCAST_SYNC_SIZE);
    for (int i=0; i< _SHMEM_BCAST_SYNC_SIZE; i+=1) {
        pSync[i] = _SHMEM_SYNC_VALUE;
    }

    parConfig * par_mat = new parConfig[1];
    local_b_buf = CollectiveInitialize(par_mat, pMatrix, pConf);    
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

    shmem_barrier_all();
    int cell_size = par_mat->div_fac*(par_mat->div_fac);
// Create localbufer for part of vec_a
    local_a_buf = (float *) shmem_malloc((par_mat->mat_fraction)*sizeof(float));
    tempbuf = (float *) shmem_malloc((par_mat->mat_fraction)*sizeof(float));
// Create vec_c data structurue
    local_c_buf = new float[cell_size];
    if (peid == 0)
        vec_a = pMatrix->mpMatrix.elements;

    for (int i=0; i<numpes; i++) {
// Distribute vec_a
        if (peid == 0) {
            memcpy(local_a_buf, (vec_a+i*(par_mat->mat_fraction)),
                (par_mat->mat_fraction*sizeof(float)));
            memcpy(tempbuf, (vec_a+i*(par_mat->mat_fraction)),
                (par_mat->mat_fraction*sizeof(float)));
        }
        shmem_barrier_all();
        shmem_broadcast32(local_a_buf, tempbuf, par_mat->mat_fraction,
            0, 0, 0, numpes, pSync);
/**
* Initialize the cuda part code
**/
        CHK_ERR(    CudaInitialize(local_a_buf, local_b_buf,
            local_c_buf, par_mat, peid, numpes));
        CHK_ERR(    ComputeMatrixAttributes(num_blocks, num_threads));
        shmem_barrier_all();
/*        if (!peid && showmat)
            rlMatrix->showmatrix(result);*/
        CHK_ERR(    CudaClean());
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
    if (!peid) {
        //rlMatrix->SaveToFile("oshm_cuda.bin");
        cout << "CleanUp!" << endl;\
        //delete plMatrix;
    }
    //delete shMatrix;
    //delete rlMatrix;

}

