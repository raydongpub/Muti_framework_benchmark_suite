#include "matdataset.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HELP_L    "--help"
#define HELP_S    "-h"

#define MAT_NUM_L "--number"
#define MAT_NUM_S "-n"

#define SHOW_MATRIX_L "--show"
#define SHOW_MATRIX_S "-s"

using namespace std;

int   mat_num = 1;
int * dim_num;
bool  showmat = false;

MatrixDataset * pMatrixset;

void getConfig(int argc, char ** argv) {

    int count = 0;

    if (argc == 1){
        cout << "\n==== HELP ====\n-h or --help\tfor help\n-n or --number [dimension_mat1";
        cout << "dimension_mat2...]\tto specify number of matrix and dimension of each matrix\n\n";
        exit(-1);
    }

    for (int i = 1; i < argc; i++){
        if ( !strcmp(argv[i], HELP_S) || !strcmp(argv[i], HELP_L) ) {
            cout << "\n==== HELP ====\n-h or --help\tfor help\n-n or --number [dimension_mat1";
            cout << "dimension_mat2...]\tto specify number of matrix and dimension of each matrix\n\n";
            exit(-1);
        }
        else if ( !strcmp(argv[i], SHOW_MATRIX_L) || !strcmp(argv[i], SHOW_MATRIX_S) ) {
            showmat = true;
        }
        else if ( !strcmp(argv[i], MAT_NUM_L) || !strcmp(argv[i], MAT_NUM_S) ) {
            mat_num = atoi(argv[++i]);
            dim_num = new int [mat_num];
            int ag  = 3;
            if (showmat)
                ag = 4;
            else
                ag = 3;
            if ((mat_num + ag) < argc || (mat_num + ag) > argc) {
                cout << "Matrix Numbers and dimension numbers are not match!" << endl;
                exit(-1);
            }
            else {
                for (int j = 0; j < mat_num; j++) {
                    count++;
                    dim_num[j] = atoi(argv[++i]);
                }
            }
        }

    }
}


int main(int argc, char ** argv) {
    getConfig(argc, argv);
    
    MatrixDataset::Matrix matrix;
    int numElements, i;    
    char filename[32];
    bool result = false;

    srand(time(NULL));
    for (i = 0; i < mat_num; i++) {
        numElements     = 2 * dim_num[i] * dim_num[i];
        pMatrixset      = new MatrixDataset();
        matrix.width    = dim_num[i];
        matrix.height   = dim_num[i];
        matrix.elements = new PRECISION[numElements];
        for (int j = 0; j < numElements; j++) {
            matrix.elements[j] = ((PRECISION) (rand()%9)) + 1.0;
        }
 
        snprintf(filename, sizeof(char) * 32, "dset%d.nn", i);
        pMatrixset->Addelements(matrix, numElements);
        pMatrixset->SaveToFile(filename);
        if (showmat)
            pMatrixset->showmatrix(result);        

        cout << "Matrix " << (i+1) << " created Successfully: ";
        cout << dim_num[i] << " x " << dim_num[i] << endl;
        memset(filename, 0, 32 * sizeof(char));
        delete    pMatrixset;
        delete [] matrix.elements; 
    }
    
    return 0;
}
