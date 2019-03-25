#include "commonBMT.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdio.h>

using namespace std;

Matrix::Matrix() {
    mpVal     = NULL;
    mpDim     = NULL;
    mpDimAddr = NULL;
    mDim      = 0;
}

Matrix::Matrix(int dim_0, int dim_1, int dim_2) {
    mpVal     = NULL;
    mpDim     = new int [3];
    mpDimAddr = NULL;
    mDim      = 3;
    count     = dim_0 * dim_1 * dim_2;

    mpDim[0] = dim_0;
    mpDim[1] = dim_1;
    mpDim[2] = dim_2;
    Allocate();
}

Matrix::Matrix(int dim_0, int dim_1, int dim_2, int dim_3) {
    mpVal     = NULL;
    mpDim     = new int [4];
    mpDimAddr = NULL;
    mDim      = 4;
    count     = dim_0 * dim_1 * dim_2 * dim_3;

    mpDim[0] = dim_0;
    mpDim[1] = dim_1;
    mpDim[2] = dim_2;
    mpDim[3] = dim_3;

    Allocate();
}

Matrix::~Matrix() {

    if (mpDimAddr != NULL)
        Deallocate();
//        delete [] mpDimAddr;

    if (mpDim != NULL)
        delete [] mpDim;

    if (mpVal != NULL)
        delete [] mpVal;
}

Matrix::PtrObj * Matrix::AllocateDimAddr(int block, int vdim) {

    PtrObj * tPtr;
    int      idx, cnt;
    int      dim = vdim + 1;
    if (dim < mDim - 1) {
        tPtr = reinterpret_cast<PtrObj *> (new PtrObj [mpDim[dim]]);
        for (idx=0;idx<mpDim[dim];idx++) {
            mpBlock[dim] = idx;
            tPtr[idx] = AllocateDimAddr(idx, dim);
        }
    }
    else {
        mpBlock[dim] = block;

        int offset = 0;
        for (int iIdx=0;iIdx<mDim - 1;iIdx++) {

            cnt = 1;
            for (int jIdx=iIdx + 1;jIdx<mDim;jIdx++) {
                cnt *= mpDim[jIdx];
            } 

            offset += mpBlock[iIdx] * cnt;
        }
        tPtr = reinterpret_cast<PtrObj *> (&mpVal[offset]);
    }
    return tPtr;
}

/*PRECISION *** Matrix::AllocateDimGddr(int block, int vdim) {

    PRECISION *** tGtr;
    tGtr = (PRECISION***)shmalloc(sizeof(PRECISION**)*mpDim[vdim+1]);
    for (int i = 0; i < mpDim[vdim+1]; i++) {
        mpBlock[vdim+1] = i;
        tGtr[i] = (PRECISION**)shmalloc(sizeof(PRECISION*)*mpDim[vdim+2]);
        for (int j = 0; j < mpDim[vdim+2]; j++) {
            mpBlock[vdim+2] = j;
            tGtr[i][j] = (PRECISION*)shmalloc(sizeof(PRECISION)*mpDim[vdim+3]);
        }
    }*/
    /*for (int i = 0; i < mpDim[vdim+1]; i++) {
        for (int j = 0; j < mpDim[vdim+2]; j++) {
            tGtr[i][j] = (PRECISION*)shmalloc(sizeof(PRECISION)*mpDim[vdim+3]);
        }
    }
    return tGtr;
}*/

void Matrix::Deallocate() {
    DeallocateDimAddr(NULL, 0);
    delete [] mpBlock;
}

void Matrix::DeallocateDimAddr(PtrObj * ptrobj, int dim) {

        PtrObj * tPtr = (dim == 0) ? reinterpret_cast<PtrObj *> (mpDimAddr) :
                                     ptrobj;

        if (dim < mDim - 2) {
            for (int blockIdx=0;blockIdx<mpDim[dim];blockIdx++) {
                    DeallocateDimAddr(reinterpret_cast<PtrObj *>(tPtr[blockIdx]),
                        dim + 1);
            }
            delete [] tPtr;
        }
        else {
            delete [] tPtr;
        }
}

void Matrix::Allocate() {

    int iIdx, cnt = 1;
    for (iIdx=0;iIdx<mDim;iIdx++)
        cnt *= mpDim[iIdx];
    mpVal = new PRECISION [cnt];
    memset(mpVal, 0, cnt * sizeof(PRECISION));

    mpBlock   = new int [mDim];
    mpDimAddr = AllocateDimAddr(0, -1);
}

PRECISION & Matrix::operator () (int dim_0, int dim_1, int dim_2) {
#if 0
    int offset = (dim_0 * mpDim[1] * mpDim[2]) +
                 (dim_1 * mpDim[2]) + 
                 dim_2;

    return mpVal[offset];
#else
    PRECISION *** ptr;
    ptr = reinterpret_cast<PRECISION ***>(mpDimAddr);
    return ptr[dim_0][dim_1][dim_2];
#endif
}

PRECISION & Matrix::operator () (int dim_0, int dim_1, int dim_2, int dim_3) {
#if 0
    int offset = (dim_0 * mpDim[1] * mpDim[2] * mpDim[3]) +
                 (dim_1 * mpDim[2] * mpDim[3]) +
                 (dim_2 * mpDim[3]) +
                 dim_3;

    return mpVal[offset];
#else 
   PRECISION **** ptr = reinterpret_cast<PRECISION ****>(mpDimAddr);
    return ptr[dim_0][dim_1][dim_2][dim_3]; 
#endif
}

PRECISION *** Matrix::GetPtr3D() {
    return reinterpret_cast<PRECISION ***>(mpDimAddr);
}

PRECISION **** Matrix::GetPtr4D() {
    return reinterpret_cast<PRECISION ****>(mpDimAddr);
}

PRECISION * Matrix::GetPtr1D() {
    return reinterpret_cast<PRECISION *>(mpVal);
}

int bmtInitMt(
    Matrix & a,     Matrix & b,     Matrix & c,
    Matrix & p,     Matrix & wrk1,  Matrix & wrk2,
    Matrix & bnd,   int      mx,    int      it,
    int      mimax, int      mjmax, int      mkmax,
    int      imax,  int      jmax,  int      kmax) {

    for (int i=0;i<mimax;++i) {
        for (int j=0;j<mjmax;++j) {
            for (int k=0;k<mkmax;++k) {
                a(0, i, j, k) = 0.0;
                a(1, i, j, k) = 0.0;
                a(2, i, j, k) = 0.0;
                a(3, i, j, k) = 0.0;
                b(0, i, j, k) = 0.0;
                b(1, i, j, k) = 0.0;
                b(2, i, j, k) = 0.0;
                c(0, i, j, k) = 0.0;
                c(1, i, j, k) = 0.0;
                c(2, i, j, k) = 0.0;

                p(i, j, k)    = 0.0;

                wrk1(i, j, k) = 0.0;
                wrk2(i, j, k) = 0.0;
                bnd(i, j, k)  = 0.0;
            }
        }
    }

    for (int i=0;i<imax;++i) {
        for (int j=0;j<jmax;++j) {
            for (int k=0;k<kmax;++k) {
                a(0, i, j, k) = 1.0;
                a(1, i, j, k) = 1.0;
                a(2, i, j, k) = 1.0;
                a(3, i, j, k) = 1.0 / 6.0;
                b(0, i, j, k) = 0.0;
                b(1, i, j, k) = 0.0;
                b(2, i, j, k) = 0.0;
                c(0, i, j, k) = 1.0;
                c(1, i, j, k) = 1.0;
                c(2, i, j, k) = 1.0;

                p(i, j, k) = (PRECISION) ((i+it) * (i+it)) /
                             (PRECISION) ((mx-1) * (mx-1));

                wrk1(i, j, k) = 0.0;
                wrk2(i, j, k) = 0.0;
                bnd(i, j, k)  = 1.0;
            }
        }
    }

    return 0;
}

