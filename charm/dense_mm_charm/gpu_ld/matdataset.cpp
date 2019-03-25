#include "matdataset.h"
#include <iostream>
#include <string.h>
#include <stdlib.h>
#ifdef _OSHM_MOD
#include <shmem.h>
#endif

using namespace std;

MatrixDataset::MatrixDataset() {
    oshm              = false;
    mNumelements      = NULL;
    mpFilename        = (char *) NULL;
    mDatasetFileSize = NULL;
}

MatrixDataset::MatrixDataset(const char * filename) {
    DataFileHeader header;

    mDatasetFile.open(filename, ios_base::in | ios_base::binary);
    if (!mDatasetFile.is_open()) {
        cout << "File open Failed!" << endl;
        exit(-1);
    }

    mDatasetFile.seekg(0, ios_base::end);
    mDatasetFileSize = mDatasetFile.tellg();
    mDatasetFile.seekg(0, ios_base::beg);

    mDatasetFile.read((char *) &header, sizeof(DataFileHeader));
    mNumelements    = header.numelements;
    mpMatrix.width  = header.width;
    mpMatrix.height = header.height;

    if ((mDatasetFileSize - sizeof(DataFileHeader)) !=
        (mNumelements * sizeof(PRECISION))) {
        cout << "Data file size do not match!" << endl;
        exit(-1);
    }
        mpMatrix.elements = (PRECISION *) malloc(mNumelements * sizeof(PRECISION));
    if (mpMatrix.elements == NULL) {
        cout << "Memory allocation Failed!" << endl;
        exit(-1);
    }

    mDatasetFile.read((char *) mpMatrix.elements, mNumelements * sizeof(PRECISION));
    mDatasetFile.close();
    mpFilename = const_cast<char *>(filename);

}


MatrixDataset::MatrixDataset(int numelements, int width, int height, bool osh) {
    oshm   = osh;
    int rc = CreateEmpty(numelements, width, height, osh);
    if (rc) {
        cout << "Create empty instance Failed!" << endl;
        exit(-1);
    }
}


MatrixDataset::~MatrixDataset() {

}

int MatrixDataset::CreateEmpty(int num, int wid, int heg, bool osh) {

    if (!oshm) {
        mpMatrix.elements = (PRECISION *) malloc(num * sizeof(PRECISION));
    }
#ifdef _OSHM_MOD
    else {
        mpMatrix.elements = (PRECISION *) shmalloc(num * sizeof(PRECISION));
    }
#endif //_OSHM_MOD
    if (mpMatrix.elements == NULL) {
        cout << "Allocate memory Failed!" << endl;
        return 1;
    }

    memset(mpMatrix.elements, 0, num * sizeof(PRECISION));
    mNumelements   = num;
    mpMatrix.width  = wid;
    mpMatrix.height = heg;

    return 0;
}

int MatrixDataset::Addelements(Matrix matrix, int numElements) {
    mpMatrix.width = matrix.width;
    mpMatrix.height = matrix.height;
    mNumelements    = numElements;
    mpMatrix.elements = (PRECISION *) new PRECISION[numElements * sizeof(PRECISION)];
    memcpy(mpMatrix.elements, matrix.elements, numElements * sizeof(PRECISION));    

    return 0;
}

int MatrixDataset::SaveToFile(const char * filename) {
    DataFileHeader header;

    mDatasetFile.open(filename, ios_base::out | ios_base::binary);
    if (!mDatasetFile.is_open()) {
        cout << "File open Failed!" << endl;
        exit(-1);
    }

    header.magic        = DS_MAGIC;
    header.version      = DS_VERSION;
    header.numelements  = mNumelements;
    header.width        = mpMatrix.width;
    header.height       = mpMatrix.height;
    mDatasetFile.write((char *) &header, sizeof(DataFileHeader));
    mDatasetFile.write((char *) mpMatrix.elements, mNumelements * sizeof(PRECISION));
    mDatasetFile.close();

    mpFilename = const_cast<char *>(filename);

    return 0;

}

int MatrixDataset::SaveToFile() {
    DataFileHeader header;

    mDatasetFile.open(mpFilename, ios_base::out | ios_base::binary);
    if (!mDatasetFile.is_open()) {
        cout << "File open Failed!" << endl;
        exit(-1);
    }

    header.magic        = DS_MAGIC;
    header.version      = DS_VERSION;
    header.numelements  = mNumelements;
    header.width        = mpMatrix.width;
    header.height       = mpMatrix.height;
    mDatasetFile.write((char *) &header, sizeof(DataFileHeader));
    mDatasetFile.write((char *) mpMatrix.elements, mNumelements * sizeof(PRECISION));
    mDatasetFile.close();

    return 0;


}

void MatrixDataset::showmatrix(bool result) {
    cout << endl << endl;
    cout <<"Matrix size is: " << mpMatrix.width << "x" << mpMatrix.height << endl;
    if (mpMatrix.elements == NULL) {
        cout << "Empty Matrix!" << endl;
    }
    else {
        if (result) {
            cout << "Matrix result is: " << endl;
        }
        else {
            cout << "Matrix 1 is: " << endl;
        }
        for (int i = 0; i < mNumelements; i++) {
            if ((i != 0) && (i % mpMatrix.width == 0)) {
                cout << endl;
            }
            if (!result && (i == mNumelements / 2)) {
                cout << "Matrix 2 is: " << endl;
            }
            cout << mpMatrix.elements[i] << " ";
    
        }
    }
    cout << endl << endl << endl;

};



