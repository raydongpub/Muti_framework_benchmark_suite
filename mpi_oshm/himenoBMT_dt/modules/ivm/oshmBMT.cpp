#include "commonBMT.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "ivm.h"

using namespace std;

PRECISION **** a,    **** b,    **** c,   *** p;
PRECISION  *** wrk1,  *** wrk2,  *** bnd;

Matrix       * pa, * pb, * pc, * pp,   * pwrk1, * pwrk2, *pbnd;
int            mx,   my,   mz,   imax,   jmax,    kmax,   it;
PRECISION      omega = 0.8, * gosa, * wgosa, * pWrk;

typedef struct {
    int l;
    int r;
} Neighbor;
typedef struct {
    Neighbor x;
    Neighbor y;
    Neighbor z;
} Cart_Neighbor;

int           numpes, peid, cartid[3];
Cart_Neighbor nb;

//Initialize and assign ID for PEs according to cartesian coordiante.
int bmtInitComm(int ndx, int ndy, int ndz) {
    if ((ndx * ndy * ndz) != numpes) {
        if (!peid) {
            cerr << endl << "Error: Number of PEs in all dimensions do not "
               << "match total number of PEs." << endl << endl;
        }
    }


    int temz, temy, temx; 
    peid = _my_pe();
    temx = peid / (ndy * ndz);
    temy = (peid - temx * ndy * ndz) / ndz;
    temz = (peid - temx * ndy * ndz - temy * ndz) % ndz;
    cartid[0] = temx;
    cartid[1] = temy;
    cartid[2] = temz;
    if (ndx > 1) {
        if (temx - 1 < 0)
            nb.x.l = -1;
        else
            nb.x.l = ((temx - 1) * ndy * ndz) + (temy * ndz) + temz;
        if (temx + 1 < ndx)
            nb.x.r = ((temx + 1) * ndy * ndz) + (temy * ndz) + temz;
        else 
            nb.x.r = -1;    
        printf("nb.x.l and nb.x.r is: %d and %d\n",nb.x.l, nb.x.r);
    }

    if (ndy > 1) {
        if (temy - 1 < 0)
            nb.y.l = -1;
        else
            nb.y.l = (temx * ndy * ndz) + ((temy - 1) * ndz) + temz;
        if (temy + 1 < ndy)
            nb.y.r = (temx * ndy * ndz) + ((temy + 1) * ndz) + temz;
        else
            nb.y.r = -1;
        printf("nb.y.l and nb.y.r is: %d and %d\n", nb.y.l, nb.y.r);
    }

    if (ndz > 1) {
        if (temz - 1 < 0)
            nb.z.l = -1;
        else
            nb.z.l = (temx * ndy * ndz) + (temy * ndz) + (temz - 1);
        if (temz + 1 < ndz)
            nb.z.r = (temx * ndy * ndz) + (temy * ndz) + (temz + 1);
        else
            nb.z.r = -1;
        printf("nb.z.l and nb.z.r is: %d and %d\n", nb.z.l, nb.z.r);
    }
    return 0;
}

//Work division and assignment for PEs
int bmtInitMax(int lmx, int lmy, int lmz) {

    int * mx1, * my1, * mz1;
    int * mx2, * my2, * mz2;
    int   tmp;

    mx1 = new int [config.mx0 + 1];
    my1 = new int [config.my0 + 1];
    mz1 = new int [config.mz0 + 1];

    mx2 = new int [config.mx0 + 1];
    my2 = new int [config.my0 + 1];
    mz2 = new int [config.mz0 + 1];

    tmp = mx / config.ndx0;
    mx1[0] = 0;
    for (int i=1;i<=config.ndx0;i++) {
        if (i <= mx % config.ndx0)
            mx1[i] = mx1[i - 1] + tmp + 1;
        else
            mx1[i] = mx1[i - 1] + tmp;
    }

    tmp = my / config.ndy0;
    my1[0] = 0;
    for (int i=1;i<=config.ndy0;i++) {
        if (i <= my % config.ndy0)
            my1[i] = my1[i - 1] + tmp + 1;
        else
            my1[i] = my1[i - 1] + tmp;
    }

    tmp = mz / config.ndz0;
    mz1[0] = 0;
    for (int i=1;i<=config.ndz0;i++) {
        if (i <= mz % config.ndz0)
            mz1[i] = mz1[i - 1] + tmp + 1;
        else
            mz1[i] = mz1[i - 1] + tmp;
    }

    //************************************************************************

    for(int i=0;i<config.ndx0;i++) {
        mx2[i] = mx1[i+1] - mx1[i];
        if(i != 0)
            mx2[i] = mx2[i] + 1;
        if(i != config.ndx0-1)
            mx2[i] = mx2[i] + 1;
    }

    for(int i=0;i<config.ndy0;i++) {
        my2[i] = my1[i+1] - my1[i];
        if(i != 0)
            my2[i] = my2[i] + 1;
        if(i != config.ndy0-1)
            my2[i] = my2[i] + 1;
    }

    for(int i=0;i<config.ndz0;i++) {
        mz2[i] = mz1[i+1] - mz1[i];
        if(i != 0)
            mz2[i] = mz2[i] + 1;
        if(i != config.ndz0-1)
            mz2[i] = mz2[i] + 1;
    }

    //************************************************************************

    if (cartid[0] == 0)
        it = mx1[cartid[0]];
    else
        it = mx1[cartid[0]] - 1;

    imax = mx2[cartid[0]];
    jmax = my2[cartid[1]];
    kmax = mz2[cartid[2]];

    delete [] mz2;
    delete [] my2;
    delete [] mx2;

    delete [] mz1;
    delete [] my1;
    delete [] mx1;

    return 0;
}

int bmtExchange_jkPlane() {

    PRECISION * tempone , * temptwo;
    int row, col;

    int size  = jmax * kmax;
    int wsize = jmax * kmax * sizeof(PRECISION); 
    tempone   = (PRECISION *)shmalloc(wsize);
    temptwo   = (PRECISION *)shmalloc(wsize);
// Pack the 3-dimension data into 1-dimension      
    if (nb.x.l > -1) {
        for (int j = 0 ; j < jmax; j++) {
            for (int k = 0; k < kmax;k++) {
                tempone[j*kmax + k] = p[1][j][k];  
            }
        }
    }
    if (nb.x.r > -1) {
        for (int j = 0 ; j < jmax; j++) {
            for (int k = 0; k < kmax;k++) {
                temptwo[j*kmax + k] = p[imax-2][j][k];
            }
        }
    }
// Synchronization to make sure all data are packed
    shmem_barrier_all();

#ifndef _DOUBLE_PRECISION
    if (nb.x.r > -1) {
// Data exchange using openshmem call (time-consuming)
        shmem_float_get(tempone, tempone, size, nb.x.r);
// Unpack the 1-dimension data into 3-dimension
        for (int i = 0; i < size; i++) {
            row = i / kmax;
            col = i % kmax;
            p[imax-1][row][col] = tempone[i];
        }
    }

    if (nb.x.l >-1) {
        shmem_float_get(temptwo , temptwo, size, nb.x.l);
        for (int i = 0; i < size; i++) {
            row = i / kmax;
            col = i % kmax;
            p[0][row][col] = temptwo[i];
        }
    }
#else
    if (nb.x.r > -1) {
        shmem_double_get(tempone, tempone, size, nb.x.r);
        for (int i = 0; i < size; i++) {
            row = i / kmax;
            col = i % kmax;
            p[imax-1][row][col] = tempone[i];
        }
    }
    if (nb.x.l > -1) {
        shmem_double_get(temptwo , temptwo, size, nb.x.l);
        for (int i = 0; i < size; i++) {
            row = i / kmax;
            col = i % kmax;
            p[0][row][col] = temptwo[i];
         }
    }
#endif
    shmem_barrier_all();
    return 0;
}

int bmtExchange_ikPlane() {

    PRECISION * tempone , * temptwo;
    int row, col;

    int size  = imax * kmax;
    int wsize = imax * kmax * sizeof(PRECISION);
    tempone   = (PRECISION *)shmalloc(wsize);
    temptwo   = (PRECISION *)shmalloc(wsize);
// Pack the 3-dimension data into 1-dimension      
    if (nb.y.l > -1) {
        for (int i = 0 ; i < imax; i++) {
            for (int k = 0; k < kmax;k++) {
                tempone[i*kmax + k] = p[i][1][k];
            }
        }
    }
    if (nb.y.r > -1) {
        for (int i = 0 ; i < imax; i++) {
            for (int k = 0; k < kmax;k++) {
                temptwo[i*kmax + k] = p[i][jmax-2][k];
            }
        }
    }
// Synchronization to make sure all data are packed
    shmem_barrier_all();

#ifndef _DOUBLE_PRECISION
    if (nb.y.r > -1) {
// Data exchange using openshmem call (time-consuming)
        shmem_float_get(tempone, tempone, size, nb.y.r);
// Unpack the 1-dimension data into 3-dimension
        for (int i = 0; i < size; i++) {
            row = i / kmax;
            col = i % kmax;
            p[row][jmax-1][col] = tempone[i];
        }
    }
    if (nb.y.l > -1) {
// Data exchange using openshmem call (time-consuming)
        shmem_float_get(temptwo, temptwo, size, nb.y.l);
// Unpack the 1-dimension data into 3-dimension
        for (int i = 0; i < size; i++) {
            row = i / kmax;
            col = i % kmax;
            p[row][0][col] = temptwo[i];
        }
    }
#else
    if (nb.y.r > -1) {
// Data exchange using openshmem call (time-consuming)
        shmem_double_get(temponw, tempone, size, nb.y.r);
// Unpack the 1-dimension data into 3-dimension
        for (int i = 0; i < size; i++) {
            row = i / kmax;
            col = i % kmax;
            p[row][jmax-1][col] = tempone[i];
        }

    }
    if (nb.y.l > -1) {
// Data exchange using openshmem call (time-consuming)
        shmem_double_get(temptwo, temptwo, size, nb.y.l);
// Unpack the 1-dimension data into 3-dimension
        for (int i = 0; i < size; i++) {
            row = i / kmax;
            col = i % kmax;
            p[row][0][col] = temptwo[i];
        }

    }

#endif
    shmem_barrier_all();
    return 0;
}

int bmtExchange_ijPlane() {

    PRECISION * tempone , * temptwo;
    int row, col;

    int size  = imax * jmax;
    int wsize = imax * jmax * sizeof(PRECISION);
    tempone   = (PRECISION *)shmalloc(wsize);
    temptwo   = (PRECISION *)shmalloc(wsize);
// Pack the 3-dimension data into 1-dimension      
    if (nb.z.l > -1) {
        for (int i = 0 ; i < imax; i++) {
            for (int j = 0; j < jmax; j++) {
                tempone[i*jmax + j] = p[i][j][1];
            }
        }
    }
    if (nb.z.r > -1) {
        for (int i = 0 ; i < imax; i++) {
            for (int j = 0; j < jmax; j++) {
                temptwo[i*jmax + j] = p[i][j][kmax-2];
            }
        }
    }
// Synchronization to make sure all data are packed
    shmem_barrier_all();

#ifndef _DOUBLE_PRECISION
    if (nb.z.r > -1) {
// Data exchange using openshmem call (time-consuming)
        shmem_float_get(tempone, tempone, size, nb.z.r);
// Unpack the 1-dimension data into 3-dimension
        for (int i = 0; i < size; i++) {
            row = i / jmax;
            col = i % jmax;
            p[row][col][kmax-1] = tempone[i];
        }

    } 
    if (nb.z.l > -1) {
// Data exchange using openshmem call (time-consuming)
        shmem_float_get(temptwo, temptwo, size, nb.z.l);
// Unpack the 1-dimension data into 3-dimension
        for (int i = 0; i < size; i++) {
            row = i / jmax;
            col = i % jmax;
            p[row][col][0] = temptwo[i];
        }

    }
#else
    if (nb.z.r > -1) {
// Data exchange using openshmem call (time-consuming)
        shmem_double_get(tempone, tempone, size, nb.z.r);
// Unpack the 1-dimension data into 3-dimension
        for (int i = 0; i < size; i++) {
            row = i / jmax;
            col = i % jmax;
            p[row][col][kmax-1] = tempone[i];
        }
    }
    if (nb.z.l > -1) {
// Data exchange using openshmem call (time-consuming)
        shmem_double_get(temptwo, temptwo, size, nb.z.l);
// Unpack the 1-dimension data into 3-dimension
        for (int i = 0; i < size; i++) {
            row = i / jmax;
            col = i % jmax;
            p[row][col][0] = temptwo[i];
        }
    }
#endif
    shmem_barrier_all();

    return 0;
}

int bmtExchange() {

    if (config.ndx0 > 1)
        bmtExchange_jkPlane();

    if (config.ndy0 > 1) {
        bmtExchange_ikPlane();
    }

    if (config.ndz0 > 1) {
        bmtExchange_ijPlane();
    }

    return 0;
}

PRECISION bmtJacobi(int nn) {

    int       n, i, j, k;
    PRECISION s0, ss;

    gosa  = (PRECISION *) shmalloc(sizeof(PRECISION));
    wgosa = (PRECISION *) shmalloc(sizeof(PRECISION));
    pSync = (long *)      shmalloc((_SHMEM_REDUCE_SYNC_SIZE)*sizeof(long));
    pWrk  = (PRECISION *) shmalloc(2 * sizeof(PRECISION));
    pWrk[0] = 1/2 + 1;
    pWrk[1] = _SHMEM_REDUCE_MIN_WRKDATA_SIZE;
    shmem_barrier_all();

    for (n=0;n<nn;++n) {

        *gosa  = 0.0;
        *wgosa = 0.0;

        for (i=1;i<imax-1;++i) {
            for (j=1;j<jmax-1;++j) {
                for (k=1;k<kmax-1;++k) {
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
                    *wgosa += ss*ss;

                    wrk2[i][j][k] = p[i][j][k] + omega * ss;
                }
            }
        }

        for (i=1;i<imax-1;++i) {
            for (j=1;j<jmax-1;++j) {
                for (k=1;k<kmax-1;++k) {
                    p[i][j][k] = wrk2[i][j][k];
                }
            }
        }

/*
** Initialize openshmem stuff
*/
        for (int i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; i++) {
            pSync[i] = _SHMEM_SYNC_VALUE;
        }
        shmem_barrier_all();
        bmtExchange();

#ifndef _DOUBLE_PRECISION
        shmem_float_sum_to_all(gosa, wgosa, 1, 0, 0, numpes, pWrk, pSync);
#else
        shmem_double_sum_to_all(gosa, wgosa, 1, 0, 0, numpes, pWrk, pSync);
#endif
//        shmem_barrier_all();

        if (!peid)
            cout << n << ": " << *gosa << endl;
    }

    return *gosa;
}

int bmtSetup(int * argc, char *** argv) {
    start_pes(0);
    numpes = _num_pes();
    peid   = _my_pe();

    mx = config.mx0 - 1; my = config.my0 - 1; mz = config.mz0 - 1;

    bool type_set = true;
    bool type_diff = false;    

    bmtInitComm(config.ndx0, config.ndy0, config.ndz0);
  
    bmtInitMax(mx, my, mz);

    pa    = new Matrix(4, config.mimax, config.mjmax, config.mkmax, type_set);
    pb    = new Matrix(3, config.mimax, config.mjmax, config.mkmax, type_set);
    pc    = new Matrix(3, config.mimax, config.mjmax, config.mkmax, type_set);
    pp    = new Matrix(config.mimax, config.mjmax, config.mkmax, type_diff);
    pwrk1 = new Matrix(config.mimax, config.mjmax, config.mkmax, type_set);
    pwrk2 = new Matrix(config.mimax, config.mjmax, config.mkmax, type_set);
    pbnd  = new Matrix(config.mimax, config.mjmax, config.mkmax, type_set);

    bmtInitMt(*pa,   *pb,    *pc, 
              *pp,   *pwrk1, *pwrk2, 
              *pbnd,  mx,     it,
               config.mimax,  config.mjmax, config.mkmax,
               imax,   jmax,  kmax);

    a    = pa->GetPtr4D();
    b    = pb->GetPtr4D();
    c    = pc->GetPtr4D();
    p    = pp->GetPtr3D();
    wrk1 = pwrk1->GetPtr3D();
    wrk2 = pwrk2->GetPtr3D();
    bnd  = pbnd->GetPtr3D();

    cout << "imax: " << imax << endl;
    cout << "jmax: " << jmax << endl;
    cout << "kmax: " << kmax << endl;
    cout << "it:   " << it   << endl;

    return 0;
}

int bmtStart() {
    shmem_barrier_all();
    PRECISION gosa = bmtJacobi(200);
    if (peid == 0)
        cout << "Gosa: " << gosa << endl;

    return 0;
}

int bmtClean() {

    delete pa;   delete pb;    delete pc;
    delete pp;   delete pwrk1; delete pwrk2;
    delete pbnd;
    shfree(gosa);
    shfree(wgosa);
    shfree(pSync);     
    shfree(pWrk);

    return 0;
}

