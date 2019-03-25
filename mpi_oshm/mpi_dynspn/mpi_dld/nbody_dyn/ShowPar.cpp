#include "ParticleDataset.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

using namespace std;

#define E_SUCCESS     0x0000
#define E_FILE_ARG    0x0001
#define E_DIFF_ARG    0x0002
#define E_TOL_ARG     0x0003
#define E_UNKNOWN_ARG 0x0004
#define E_NO_ARG      0x0005

#define MODE_HELP     0x0000
#define MODE_SHOW     0x0001
#define MODE_DIFF     0x0002

const char * e_msg[] = {
"No error",
"Input file is missing.",
"Two input files are required for diff-mode.",
"Tolerance value is missing.",
"Unknown argument found.",
"Arguments needed."
};

const char * eds_msg[] = {
"No error",
"Unknown error",
"Input file not found.",
"File I/O error",
"Memory allocation failed",
"Invalid format of input file"
};

const char * help_msg = "\
Usage: sp [OPTIONS] [FILE]                                                  \n\
Options are available in both short and long format:                        \n\
\t-d, --diff          Compare particle set in two file.                     \n\
\t    --show-all-diff Show all differences in two file.                     \n\
\t-t, --tolerance     Specify tolerance for comparison.                     \n\
\t-h, --help          Show this help message.";

int         mode        = MODE_SHOW;
char      * filename[2] = {NULL, NULL};
PRECISION   tolerance   = 0.0;
bool        showAllDiff = false;


inline int SetPreference(int argc, char ** argv) {
#define IS_OPTION(str) (!strcmp(argv[idx], str))
    bool arg = false;

    if (argc <= 1)
        return E_NO_ARG;

    int idx = 1;

    while (idx < argc) {
        if IS_OPTION("-d") {
            arg = true;
            if ((idx + 2) >= argc)
                return E_DIFF_ARG;
              mode        = MODE_DIFF;
              filename[0] = argv[++idx];
              filename[1] = argv[++idx];
        }
        else if IS_OPTION("--diff") {
            arg = true;
            if ((idx + 2) >= argc)
                return E_DIFF_ARG;
              mode        = MODE_DIFF;
              filename[0] = argv[++idx];
              filename[1] = argv[++idx];
        }
        else if IS_OPTION("-t") {
            arg = true;
            if ((idx + 1) >= argc)
                return E_TOL_ARG;
              tolerance = atof(argv[++idx]);
        }
        else if IS_OPTION("--tolerance") {
            arg = true;
            if ((idx + 1) >= argc)
                return E_TOL_ARG;
              tolerance = atof(argv[++idx]);
        }
        else if IS_OPTION("--show-all-diff") {
            arg = true;
            showAllDiff = true;
        }
        else if IS_OPTION("-h") {
            arg = true;
            mode = MODE_HELP;
            return E_SUCCESS;
        }
        else if IS_OPTION("--help") {
            arg = true;
            mode = MODE_HELP;
            return E_SUCCESS;
        }
        else {
            if (!arg) filename[0] = argv[1];
            return (arg) ? E_UNKNOWN_ARG : E_SUCCESS;
        }

        idx++;
    }

    return E_SUCCESS;

#undef IS_OPTION
}

int main(int argc, char ** argv) {

    int rc;
    if ((rc = SetPreference(argc, argv)) != E_SUCCESS) {
        cerr << endl << "Error: " << e_msg[rc] << endl << endl;
        exit(rc);
    }

    ParticleDataset * dataset[2];

    switch (mode) {
    case MODE_HELP:
        cout << endl << help_msg << endl << endl;
        break;

    case MODE_SHOW:
        try {
            dataset[0] = new ParticleDataset(filename[0]);
        } catch (int e) {
            cerr << endl << "Error: " << eds_msg[e - EDS_SUCCESS]
                << endl << endl;
            exit(1);
        }
        for (int idx=0;idx<dataset[0]->mNumParticles;idx++) {
            cout << "[" << idx << "]:\t"
                << dataset[0]->mpParticle[idx].xPos << "\t"
                << dataset[0]->mpParticle[idx].yPos << "\t"
                << dataset[0]->mpParticle[idx].zPos << endl;
        }

        delete dataset[0];
        break;

    case MODE_DIFF:
        for (int idx=0;idx<2;idx++) {
            try {
                dataset[idx] = new ParticleDataset(filename[idx]);
            }
            catch (int e) {
                cerr << endl << "Error: " << eds_msg[e - EDS_SUCCESS]
                    << endl << endl;
                exit(1);
            }
        }

        if (dataset[0]->mNumParticles != dataset[1]->mNumParticles) {
            cout << "Warning: Number of particles in the datasets "
                << "do not match (" << dataset[0]->mNumParticles
                << "/" << dataset[1]->mNumParticles << ")" << endl;
        }

        int min = (dataset[0]->mNumParticles < dataset[1]->mNumParticles)
            ? dataset[0]->mNumParticles : dataset[1]->mNumParticles;

        PRECISION max_diff = 0.0, x_diff, y_diff, z_diff;
        int       pIdx     = 0;   bool diff = false;

        for (int idx=0;idx<min;idx++) {
#define CHECK_DIFF(axis)                                       \
    axis##_diff = fabs(dataset[0]->mpParticle[idx].axis##Pos - \
        dataset[1]->mpParticle[idx].axis##Pos);                \
    if (axis##_diff > tolerance) {                             \
        diff = true;                                           \
        if (axis##_diff > max_diff) {                          \
            max_diff = axis##_diff;                            \
            pIdx     = idx;                                    \
        }                                                      \
        if (showAllDiff)                                       \
            cout << axis##_diff << "\t";                       \
    }
            if (showAllDiff)
                cout << "[" << idx << "]:\t";
            CHECK_DIFF(x);
            CHECK_DIFF(y);
            CHECK_DIFF(z);
            if (showAllDiff)
                cout << endl;
#undef CHECK_DIFF
        }
        if (diff) {
            cout << endl << "Verify FAILED" << endl;
            cout << "Max diff: " << max_diff << " @ " << pIdx << endl << endl;
        }
        else {
            cout << endl << "Verify PASSED" << endl << endl;
        }

        delete dataset[0];
        delete dataset[1];
        break;
    }

    return 0;
}

