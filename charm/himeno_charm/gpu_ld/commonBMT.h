#ifndef _COMMONBMT_H_
#define _COMMONBMT_H_

#include "pup.h"

#ifndef _DOUBLE_PRECISION
#define PRECISION float
#else
#define PRECISION double
#endif

typedef struct {
    int mx0;
    int my0;
    int mz0;
    int mimax;
    int mjmax;
    int mkmax;
    int ndx0;
    int ndy0;
    int ndz0;
    void pup(PUP::er &p) {
        p|mx0;p|my0;p|mz0;
        p|mimax;p|mjmax;p|mkmax;
        p|ndx0;p|ndy0;p|ndz0;
    }

} BMT_Config;

class Matrix {
public:
    Matrix();
    Matrix(int dim_0, int dim_1, int dim_2);
    Matrix(int dim_0, int dim_1, int dim_2, int dim_3);
    ~Matrix();

    PRECISION & operator () (int dim_0, int dim_1, int dim_2);
    PRECISION & operator () (int dim_0, int dim_1, int dim_2, int dim_3);
    PRECISION ***  GetPtr3D();
    PRECISION **** GetPtr4D();
    PRECISION *    GetPtr1D();
 
    PRECISION * mpVal;
    int       * mpDim;
    int         mDim;
    int         count;

/*    void pup(PUP::er &p) {
        p|count;
        if (p.isUnpacking()) {
            mpVal = new PRECISION[count];
        }
        PUParray(p, mpVal, count); 
    }*/

private:
    typedef void * PtrObj;
    PtrObj * AllocateDimAddr(int block, int dim);
//    PRECISION *** AllocateDimGddr(int block, int dim);
    void     DeallocateDimAddr(PtrObj * ptrobj, int dim);
    void     Allocate();
    void     Deallocate();

    PtrObj * mpDimAddr;
    PRECISION *** mpDimGddr;
    int    * mpBlock;
};

//Implemented by common module
int bmtInitMt(
    Matrix & a,     Matrix & b,     Matrix & c,
    Matrix & p,     Matrix & wrk1,  Matrix & wrk2,
    Matrix & bnd,   int      mx,    int      it,
    int      mimax, int      mjmax, int      mkmax,
    int      imax,  int      jmax,  int      kmax);

//Implemented by modules
int bmtSetup(int * argc, char *** argv);
int bmtStart();
int bmtClean();

#ifndef _MAIN
extern BMT_Config config;
#endif

#endif //_COMMONBMT_H
