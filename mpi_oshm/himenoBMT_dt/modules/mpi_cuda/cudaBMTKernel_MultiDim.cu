#include "cudaBMTKernel_MultiDim.cuh"
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

//#define _DEBUG

using namespace std;

PRECISION    * fa_d,      * fb_d,      * fc_d, 
             * fp_d,      * fwrk1_d,   * fwrk2_d,  * fbnd_d,
          **** a_d,    **** b_d,    **** c_d,
           *** p_d,     *** wrk1_d,  *** wrk2_d, *** bnd_d,
             * gosa_d,
          **** a_h,    **** b_h,    **** c_h,
           *** p_h,     *** wrk1_h,  *** wrk2_h, *** bnd_h,
             * gosa_h;

typedef void * PtrObj;

__global__ void bmtJacobiKernel(
    PRECISION **** a,    PRECISION **** b,    PRECISION **** c,
    PRECISION  *** p,    PRECISION  *** wrk1, PRECISION  *** wrk2,
    PRECISION  *** bnd,  PRECISION    * gosa,
    int            imax, int            jmax, int            kmax) {

    int       i, j, k ,i_s, j_s, k_s, i_strides, j_strides, k_strides;
    PRECISION s0, ss, omega = 0.8;

//    __shared__ PRECISION wgosa;

    int boffset_x = blockIdx.x * blockDim.x;
    int boffset_y = blockIdx.y * blockDim.y;
    int boffset_z = 0;

    int totThreadsx = gridDim.x * blockDim.x;
    int gThreadIdxx = boffset_x + threadIdx.x;
    int totThreadsy = gridDim.y * blockDim.y;
    int gThreadIdxy = boffset_y + threadIdx.y;
    int totThreadsz = blockDim.z;
    int gThreadIdxz = boffset_z + threadIdx.z;

//    int tid = (threadIdx.z * (blockDim.y * blockDim.x)) +
//              (threadIdx.y * blockDim.x) +
//              threadIdx.x;

//    if (tid == 0)
//        wgosa = 0.0;
//    __syncthreads();

    i_strides = (imax / totThreadsx) + 1;
    j_strides = (jmax / totThreadsy) + 1;
    k_strides = (kmax / totThreadsz) + 1;

    for (int xxx=0;xxx<8;xxx++) {
    for (i_s=0;i_s<i_strides;i_s++) {

        i = (i_s * totThreadsx) + gThreadIdxx;
        if ((i < 1) || (i > imax - 2))
            continue;

        for(int yyy=0;yyy<8;yyy++) {
        for (j_s=0;j_s<j_strides;j_s++) {

            j = (j_s * totThreadsy) + gThreadIdxy;
            if ((j < 1) || (j > jmax - 2))
                continue;


            for (int zzz=0;zzz<8;zzz++) {
            for (k_s=0;k_s<k_strides;k_s++) {

                k = (k_s * totThreadsz) + gThreadIdxz;
                if ((k < 1) || (k > kmax - 2))
                    continue;

                s0 = a[0][i][j][k] * p[i+1][j  ][k  ]
                   + a[1][i][j][k] * p[i  ][j+1][k  ]
                   + a[2][i][j][k] * p[i  ][j  ][k+1]
                   + b[0][i][j][k] * ( p[i+1][j+1][k  ] - p[i+1][j-1][k  ]
                                     - p[i-1][j+1][k  ] + p[i-1][j-1][k  ] )
                   + b[1][i][j][k] * ( p[i  ][j+1][k+1] - p[i  ][j-1][k+1]
                                     - p[i  ][j+1][k-1] + p[i  ][j-1][k-1] )
                   + b[2][i][j][k] * ( p[i+1][j  ][k+1] - p[i-1][j  ][k+1]
                                     - p[i+1][j  ][k-1] + p[i-1][j  ][k-1] )
                   + c[0][i][j][k] * p[i-1][j  ][k  ]
                   + c[1][i][j][k] * p[i  ][j-1][k  ]
                   + c[2][i][j][k] * p[i  ][j  ][k-1]
                   + wrk1[i][j][k];

                ss = ( s0 * a[3][i][j][k] - p[i][j][k] ) * bnd[i][j][k];
                atomicAdd(gosa, ss*ss);

                wrk2[i][j][k] = p[i][j][k] + omega * ss;
            }
            }
        }
        }
    }
    }

//    __syncthreads();
/*
    for (i=1;i<imax-1;++i) {
        for (j=1;j<jmax-1;++j) {
            for (k=1;k<kmax-1;++k) {
*/
#if 0
    for (i_s=0;i_s<i_strides;i_s++) {

        i = (i_s * totThreadsx) + gThreadIdxx;
        if ((i < 1) || (i > imax - 2))
            continue;

        for (j_s=0;j_s<j_strides;j_s++) {

            j = (j_s * totThreadsy) + gThreadIdxy;
            if ((j < 1) || (j > jmax - 2))
                continue;

            for (k_s=0;k_s<k_strides;k_s++) {

                k = (k_s * totThreadsz) + gThreadIdxz;
                if ((k < 1) || (k > kmax - 2))
                    continue;

                p[i][j][k] = wrk2[i][j][k];
            }
        }
    }
#endif
#if 0
    if (tid == 0) {
        printf("gosa: %f\n", wgosa);
        atomicAdd(gosa, wgosa);
    }
#endif
}

__global__ void bmtUpdatePressureKernel(
    PRECISION *** p, PRECISION *** wrk2, int imax, int jmax, int kmax) {

    int       i, j, k ,i_s, j_s, k_s, i_strides, j_strides, k_strides;

    int boffset_x = blockIdx.x * blockDim.x;
    int boffset_y = blockIdx.y * blockDim.y;
    int boffset_z = 0;

    int totThreadsx = gridDim.x * blockDim.x;
    int gThreadIdxx = boffset_x + threadIdx.x;
    int totThreadsy = gridDim.y * blockDim.y;
    int gThreadIdxy = boffset_y + threadIdx.y;
    int totThreadsz = blockDim.z;
    int gThreadIdxz = boffset_z + threadIdx.z;

    i_strides = (imax / totThreadsx) + 1;
    j_strides = (jmax / totThreadsy) + 1;
    k_strides = (kmax / totThreadsz) + 1;

    for (i_s=0;i_s<i_strides;i_s++) {

        i = (i_s * totThreadsx) + gThreadIdxx;
        if ((i < 1) || (i > imax - 2))
            continue;

        for (j_s=0;j_s<j_strides;j_s++) {

            j = (j_s * totThreadsy) + gThreadIdxy;
            if ((j < 1) || (j > jmax - 2))
                continue;

            for (k_s=0;k_s<k_strides;k_s++) {

                k = (k_s * totThreadsz) + gThreadIdxz;
                if ((k < 1) || (k > kmax - 2))
                    continue;

                p[i][j][k] = wrk2[i][j][k];
            }
        }
    }
}

#ifdef _DEBUG
__global__ void DebugKernel(PRECISION **** a) {

    a[0][1][2][3] = 100;
    a[3][0][2][1] = 200;

}
#endif

#define CHK_ERR(str)           \
    do {                       \
        cudaError_t ce = str;  \
        if (ce != cudaSuccess) \
            return ce;         \
    } while (0)

int bmtAssign_MultiDimension_Space_Rec(
    PtrObj * ptrobj, PtrObj * ptrobj_d, PRECISION * flat_d,   int dim,
    int      mdim,  int     * adim,     int         poffset,  int doffset,
    int * blocks) {
#ifdef _DEBUG
#define INDENT for (int i=0;i<dim;i++) cout << "\t";
#endif

    int iIdx, offset = doffset;

    if (dim < mdim - 2) {
        int nloffset = 1;
        for (int idx=0;idx<=dim;idx++)
            nloffset *= adim[idx];
        nloffset += doffset;

        for (iIdx=0;iIdx<adim[dim];iIdx++) {
            blocks[dim] = iIdx;

            int loffset = 0;
            if (dim > 0) {
                int b=0;
                for (int i=0;i<dim;i++) {
                    if (i != dim - 1)
                        b += blocks[i] * adim[i+1];
                     else
                        b += blocks[i];
                }
                loffset += adim[dim] * b;
            }
#ifdef _DEBUG
            INDENT;
            cout << "[" << dim << ", " << iIdx << "]:" << adim[dim]
               << ": " << offset + loffset<< endl;
#endif

            bmtAssign_MultiDimension_Space_Rec(
                ptrobj, ptrobj_d, flat_d, dim + 1,
                mdim, adim, offset + loffset, nloffset, blocks);

            if ((poffset != -1) && (iIdx == 0))
                ptrobj[poffset] = ptrobj_d + offset + loffset;
                    /*reinterpret_cast<PtrObj>(offset+loffset);*/

            offset++;
        }
    }
    else {
        if (dim > 0) {
            int b=0;
            for (int i=0;i<dim;i++) {
                if (i != dim - 1)
                    b += blocks[i] * adim[i+1];
                 else
                    b += blocks[i];
            }
            offset += adim[dim] * b;
        }

        for (iIdx=0;iIdx<adim[dim];iIdx++) {

#ifdef _DEBUG
            INDENT;
            cout << "[" << dim << ", " << iIdx << "]:" << adim[dim]
               << ": " << offset << endl;
#endif

            if ((poffset != -1) && (iIdx == 0))
                ptrobj[poffset] = ptrobj_d + offset;
                    /*reinterpret_cast<PtrObj>(offset);*/

            int foffset = 0;
            for (int i=0;i<mdim-1;i++) {

                int ele = 1;
                for (int j=i+1;j<mdim;j++)
                   ele *= adim[j];

                if (i < mdim - 2)
                    foffset += blocks[i] * ele;
                else
                    foffset += iIdx * ele;
            }

            ptrobj[offset] = flat_d + foffset;
                /*reinterpret_cast<PtrObj>(foffset);*/

            offset++;
        }
    }

    return 0;
}

int bmtCreateDevice_MultiDimension_Space(
    PRECISION ** m_h, PRECISION ** m_d,  PRECISION * fm_d,
    int          dim, int        * adim) {

    int iIdx, jIdx, cnt = 1;

    //Determine the number of blocks for storing pointer objects
    for (iIdx=0;iIdx<dim-1;iIdx++)
        cnt *= adim[iIdx];

    for (iIdx=dim-3;iIdx>=0;iIdx--) {
        int tcnt = 1;
        for (jIdx=iIdx;jIdx>=0;jIdx--)
            tcnt *= adim[jIdx];
        cnt += tcnt;
    }
#ifdef _DEBUG
    cout << "***" << cnt << endl;
#endif
    //Allocate blocks for storing pointer objects on both host and device
    PtrObj * tm_h, * tm_d;
    tm_h  = new PtrObj[cnt];
    CHK_ERR(    cudaMalloc(&tm_d, cnt * sizeof(PtrObj)));

    //Assign pointer values to blocks
    int blocks[4];
    bmtAssign_MultiDimension_Space_Rec(
        tm_h, tm_d, fm_d, 0, dim, 
        adim, -1,   0,    blocks);

    //Transfer the created multidimentional array to device
    CHK_ERR(    cudaMemcpy(tm_d, tm_h,
        cnt * sizeof(PtrObj), cudaMemcpyHostToDevice));

    *m_h = reinterpret_cast<PRECISION *>(tm_h);
    *m_d = reinterpret_cast<PRECISION *>(tm_d);

#ifdef _DEBUG
    cout << endl << "Origin:\t" << tm_d << endl;
    for (iIdx=0;iIdx<cnt;iIdx++)
        cout << iIdx << ":\t" << tm_h[iIdx] << endl;
#endif

    return 0;
}

cudaError_t bmtInitDeviceMemory(
    Matrix * pa,   Matrix * pb,    Matrix * pc,
    Matrix * pp,   Matrix * pwrk1, Matrix * pwrk2,
    Matrix * pbnd, int      peid) {

    int devCnt = 0;
    CHK_ERR(    cudaGetDeviceCount(&devCnt));
    CHK_ERR(    cudaSetDevice(peid % devCnt));
//    CHK_ERR(    cudaSetDevice(0));

    gosa_h = new PRECISION();
    CHK_ERR(    cudaMalloc(&gosa_d, sizeof(PRECISION)));

    size_t memreq_3d = config.mimax * config.mjmax *
                       config.mkmax * sizeof(PRECISION);

    CHK_ERR(    cudaMalloc(&fa_d, 4 * memreq_3d));
    CHK_ERR(    cudaMalloc(&fb_d, 3 * memreq_3d));
    CHK_ERR(    cudaMalloc(&fc_d, 3 * memreq_3d));
    CHK_ERR(    cudaMalloc(&fp_d,     memreq_3d));
    CHK_ERR(    cudaMalloc(&fwrk1_d,  memreq_3d));
    CHK_ERR(    cudaMalloc(&fwrk2_d,  memreq_3d));
    CHK_ERR(    cudaMalloc(&fbnd_d,   memreq_3d));

    CHK_ERR(    cudaMemcpy(fa_d,    pa->mpVal,
        4 * memreq_3d, cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(fb_d,    pb->mpVal,
        3 * memreq_3d, cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(fc_d,    pc->mpVal,
        3 * memreq_3d, cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(fp_d,    pp->mpVal,
        memreq_3d, cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(fwrk1_d, pwrk1->mpVal,
        memreq_3d, cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(fwrk2_d, pwrk2->mpVal,
        memreq_3d, cudaMemcpyHostToDevice));
    CHK_ERR(    cudaMemcpy(fbnd_d,  pbnd->mpVal,
        memreq_3d, cudaMemcpyHostToDevice));

#ifndef _DEBUG

    //Construct multi-dimensional space for matrices
    bmtCreateDevice_MultiDimension_Space(
        reinterpret_cast<PRECISION **>(&a_h),
        reinterpret_cast<PRECISION **>(&a_d),
        fa_d, pa->mDim, pa->mpDim);

    bmtCreateDevice_MultiDimension_Space(
        reinterpret_cast<PRECISION **>(&b_h),
        reinterpret_cast<PRECISION **>(&b_d),
        fb_d, pb->mDim, pb->mpDim);

    bmtCreateDevice_MultiDimension_Space(
        reinterpret_cast<PRECISION **>(&c_h),
        reinterpret_cast<PRECISION **>(&c_d),
        fc_d, pc->mDim, pc->mpDim);

    bmtCreateDevice_MultiDimension_Space(
        reinterpret_cast<PRECISION **>(&p_h),
        reinterpret_cast<PRECISION **>(&p_d),
        fp_d, pp->mDim, pp->mpDim);

    bmtCreateDevice_MultiDimension_Space(
        reinterpret_cast<PRECISION **>(&wrk1_h),
        reinterpret_cast<PRECISION **>(&wrk1_d),
        fwrk1_d, pwrk1->mDim, pwrk1->mpDim);

    bmtCreateDevice_MultiDimension_Space(
        reinterpret_cast<PRECISION **>(&wrk2_h),
        reinterpret_cast<PRECISION **>(&wrk2_d),
        fwrk2_d, pwrk2->mDim, pwrk2->mpDim);

    bmtCreateDevice_MultiDimension_Space(
        reinterpret_cast<PRECISION **>(&bnd_h),
        reinterpret_cast<PRECISION **>(&bnd_d),
        fbnd_d, pbnd->mDim, pbnd->mpDim);

#else
    PRECISION **** fake_h, **** fake_d, * ffake_d;
    Matrix       * pfake;

    pfake  = new Matrix(4,2,3,4);
    CHK_ERR(    cudaMalloc(&ffake_d, 4 * 2 * 3 * 4 * sizeof(PRECISION)));
    CHK_ERR(    cudaMemset(ffake_d, 0, 4 * 2 * 3 * 4 * sizeof(PRECISION)));

    bmtCreateDevice_MultiDimension_Space(
        reinterpret_cast<PRECISION **>(&fake_h),
        reinterpret_cast<PRECISION **>(&fake_d),
        ffake_d, pfake->mDim, pfake->mpDim);

    DebugKernel <<<256, 512>>> (fake_d);
    CHK_ERR(    cudaDeviceSynchronize());

    CHK_ERR(    cudaMemcpy(pfake->mpVal, ffake_d, 4 * 2 * 3 * 4 *
        sizeof(PRECISION), cudaMemcpyDeviceToHost));
    for (int i=0;i<4;i++) {
        cout << "[0, " << i << "]" << endl;

        for (int j=0;j<2;j++) {
            cout << "\t[1, " << j << "]" << endl;

            for (int k=0;k<3;k++) {
                cout << "\t\t[2, " << k << "]" << endl;
                cout << "\t\t";
                for (int l=0;l<4;l++) {
                    cout << pfake->mpVal[(i*24)+(j*12)+(k*4)+l] << "\t";
                }
                cout << endl;
            }
            cout << endl;
        }
    }

#endif

    return cudaSuccess;
}

cudaError_t bmtCudaJacobi(PRECISION * gosa, Matrix * pp,
    int imax, int jmax, int kmax, int num_blocks, int num_threads) {

    dim3 grid(num_blocks, num_blocks, 1); // grid(16, 16, 1)
    dim3 block(1, 1, num_threads); // block(1, 1, 64);

    size_t memreq_3d = config.mimax * config.mjmax *
                       config.mkmax * sizeof(PRECISION);

    struct timeval s_tv, e_tv;

//    for (int idx=0;idx<nn;idx++) {
        //Jacobi
        CHK_ERR(    cudaMemset(gosa_d, 0, sizeof(PRECISION)));

        gettimeofday(&s_tv, NULL);

        bmtJacobiKernel <<<grid, block>>> (
            a_d,  b_d,  c_d, p_d, wrk1_d, wrk2_d, bnd_d, gosa_d,
            imax, jmax, kmax);
        CHK_ERR(    cudaDeviceSynchronize());

        //Update Pressure Matrix
        bmtUpdatePressureKernel <<<grid, block>>> (
            p_d,  wrk2_d,
            imax, jmax, kmax);
        CHK_ERR(    cudaDeviceSynchronize());

        gettimeofday(&e_tv, NULL);
        double start = ((double) s_tv.tv_sec * 1000000.0) +
            (double) s_tv.tv_usec;
        double end   = ((double) e_tv.tv_sec * 1000000.0) +
            (double) e_tv.tv_usec;
        double time_used = (end - start) / 1000000.0;
        cout << "Kernel time: " << time_used << endl;


        CHK_ERR(    cudaMemcpy(gosa_h, gosa_d,
            sizeof(PRECISION), cudaMemcpyDeviceToHost));

        CHK_ERR(    cudaMemcpy(pp->mpVal, fp_d,
            memreq_3d, cudaMemcpyDeviceToHost));

        *gosa = *gosa_h;

//        cout << idx << ": " << *gosa_h << endl;
//    }
//    CHK_ERR(    cudaMemcpy(gosa_h, gosa_d,
//        sizeof(PRECISION), cudaMemcpyDeviceToHost));

    return cudaSuccess;
}

