#ifndef _MAIN_H_
#define _MAIN_H_

#include <matdataset.h>

typedef struct {
    int mrank;
    int mcommSize;
} pConfig;

typedef struct {
    int beg;
    int div_fac;
    int mat_fraction;
    int wid;
    int heg;
    int num;
    PRECISION * mat_cell;
} parConfig;

extern "C" {
int LibSetup(MatrixDataset * pMatrix, pConfig * pConf);

int LibEntry(bool showmat, int num_blocks, int num_threads, MatrixDataset * pMatrix, pConfig * pCon);

int LibCleanUp(void);
}

#endif
