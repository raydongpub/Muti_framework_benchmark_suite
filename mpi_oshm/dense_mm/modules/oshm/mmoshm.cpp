#include "main.h"
//#include "matdataset.h"
#include "mpi.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <shmem.h>
#include <string.h>

using namespace std;

MatrixDataset * plMatrix;
MatrixDataset * rlMatrix;
MatrixDataset * shMatrix;

int numpes, peid;
int *Numelements, numelements, *wid, *heg;

int LibSetup(MatrixDataset * pMatrix, pConfig * pConf) {
    numpes   = pConf->mcommSize;
    peid     = pConf->mrank;
    if (!peid)
        plMatrix = pMatrix;
    return 0;
}

inline void CollectiveInitialize(int * startpoint, int * numpoint) {

    Numelements = (int *) shmalloc(sizeof(int));
    wid         = (int *) shmalloc(sizeof(int));
    heg         = (int *) shmalloc(sizeof(int));

    if (!peid){
        *Numelements = plMatrix->mNumelements;
        *wid         = plMatrix->mpMatrix.width;
        *heg         = plMatrix->mpMatrix.height;
    }
/**
* Pass the variale
**/
    shmem_barrier_all();
    if (peid) {
        shmem_int_get(Numelements, Numelements, 1, 0);
        shmem_int_get(wid, wid, 1, 0);
        shmem_int_get(heg, wid, 1, 0);
    }

    numelements = *Numelements / 2;
/**
* Allocate shared space for all PE
**/
    int oshm = true;
    shMatrix = new MatrixDataset(*Numelements, *wid, *heg, oshm);
    rlMatrix = new MatrixDataset(numelements, *wid, *heg, oshm);
     
    if (!peid) {
        memcpy(shMatrix->mpMatrix.elements, plMatrix->mpMatrix.elements, *Numelements * sizeof(PRECISION));
    }
    shmem_barrier_all();    
    if (peid) {
#ifndef _DOUBLE_PRECISION
        shmem_float_get(shMatrix->mpMatrix.elements, shMatrix->mpMatrix.elements, *Numelements, 0);
#else
        shmem_double_get(shMatrix->mpMatrix.elements, shMatrix->mpMatrix.elements, *Numelements, 0);
#endif
    }
/**
* Partitioned the work
**/
   int divCnt = numelements / numpes;
   int remCnt = numelements % numpes;
   if (peid < remCnt) {
       *numpoint   = divCnt + 1;
       *startpoint = peid * (*numpoint);
   }
   else {
       *numpoint = divCnt;
       *startpoint = remCnt + peid * divCnt;
   }

}

int LibEntry(bool showmat, num_blocks, num_threads) {
    
    int col,row;
    bool oshm = true, result = true;
    int mstart, mnum;
    CollectiveInitialize(&mstart, &mnum);    
    shmem_barrier_all();
/**
* Computation begin
**/
    for (int i = 0; i < mnum; i++) {
        for (int j = 0; j < *wid; j++) {
            row = (i+mstart) / *wid;
            col = (i+mstart) % *wid;
            rlMatrix->mpMatrix.elements[i+mstart] +=
                shMatrix->mpMatrix.elements[row*(*wid) + j] * shMatrix->mpMatrix.elements[numelements + col + j*(*wid)];
        }
    }

/**
* Collective merge data back to head node
**/
    if (peid) {
        shmem_putmem((rlMatrix->mpMatrix.elements + mstart), (rlMatrix->mpMatrix.elements + mstart),
            mnum * sizeof(PRECISION), 0);
    }
    shmem_barrier_all();
    if (!peid && showmat)
        rlMatrix->showmatrix(result);

}

int LibCleanUp(void) {
    if (!peid) {
        rlMatrix->SaveToFile("oshm.bin");
        cout << "CleanUp!" << endl;\
        delete plMatrix;
    }
    delete shMatrix;
    delete rlMatrix;
    shfree(Numelements);
    shfree(wid);
    shfree(heg);

    return 0;
}

