#include "main.h"
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;

MatrixDataset * plMatrix;
MatrixDataset * rlMatrix;

int LibSetup(MatrixDataset * pMatrix, pConfig * pConf) {
    plMatrix = pMatrix;
}

int LibEntry(bool showmat, int num_blocks, int num_threads) {
    int Numelements = plMatrix->mNumelements;
    int numelements = Numelements / 2;
    int wid         = plMatrix->mpMatrix.width, heg = plMatrix->mpMatrix.height;
    int col,row;
    bool oshm       = false, result = true;

    rlMatrix = new MatrixDataset(numelements, wid, heg, oshm);
/**
* Computation begin
**/
    for (int i = 0; i < numelements; i++) {
        for (int j = 0; j < wid; j++) {
            row = i / wid;
            col = i % wid;
            rlMatrix->mpMatrix.elements[i] += 
                plMatrix->mpMatrix.elements[row*wid + j] * plMatrix->mpMatrix.elements[numelements + col + j*wid];
        }
    }
    if (showmat)
        rlMatrix->showmatrix(result);
    
}

int LibCleanUp(void) {
    rlMatrix->SaveToFile("st.bin");
    cout << "CleanUp!" << endl;
}
