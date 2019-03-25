#define _MAIN

#include "commonBMT.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#ifdef _OSHM_MOD
#include <shmem.h>
#endif //_OSHM_MOD
#include <sys/time.h>
#include <math.h>

using namespace std;

#define MODE_NORM 0x0000
#define MODE_HELP 0x0001

#define DSET_XS   0x0000
#define DSET_S    0x0001
#define DSET_M    0x0002
#define DSET_L    0x0003
#define DSET_XL   0x0004

#define E_SUCCESS 0x0000
#define E_NO_ARG  0x0001
#define E_UNKNOWN 0x0002
#define E_INV_PE  0x0003
#define E_INV_PEV 0x0004
#define E_INV_DS  0x0005
#define E_INV_DSV 0x0006

const char * e_msg[] = {
"No error",
"No arguments specified.",
"Unrecognized arguments presented.",
"Requires three PE numbers along the dimensions.",
"Invalid PE numbers specified.",
"Requires the size of dataset (xs, s, m, l, xl).",
"Unrecognized dataset size"
};

const char h_msg[] = {"\
Usage: %s [OPTIONS] [...]                                                   \n\
Options are available in both short and long format:                      \n\n\
\t-pe [pe_x pe_y pe_z]     Specify numbers of PEs along dimensions          \n\
\t-h, --help               Show this help message.                          \n\
"};

int           gargc;
char       ** gargv;
int           mode = MODE_NORM;
BMT_Config    config;
int           blocks, threads;

inline int SetPreference(int argc, char ** argv) {
#define IS_OPTION(str) (!strcmp(argv[idx], str))

    gargc = argc; gargv = argv;

    int idx = 1;

    if (argc < 2)
        return E_NO_ARG;

    while (idx < argc) {
        if IS_OPTION("-pe") {
            if ((idx + 3) >= argc)
                return E_INV_PE;
             int temp = atoi(argv[++idx]);
             if (temp < 1)
                 return E_INV_PEV;
             config.ndx0 = temp;

             temp = atoi(argv[++idx]);
             if (temp < 1)
                 return E_INV_PEV;
             config.ndy0 = temp;

             temp = atoi(argv[++idx]);
             if (temp < 1)
                 return E_INV_PEV;
             config.ndz0 = temp;
        }
        else if (IS_OPTION("-b")) {
            blocks = atoi(argv[++idx]);
        }
        else if (IS_OPTION("-t")) { 
            threads = atoi(argv[++idx]);
        }
        else if (IS_OPTION("-ds") || IS_OPTION("--dataset-size")) {
            if ((idx + 1) >= argc)
                return E_INV_DS;
            idx++;
            if IS_OPTION("xs") {
                config.mx0 = 33;
                config.my0 = 33;
                config.mz0 = 65;
            }
            else if IS_OPTION("s") {
                config.mx0 = 65;
                config.my0 = 65;
                config.mz0 = 129;
            }
            else if IS_OPTION("m") {
                config.mx0 = 129;
                config.my0 = 129;
                config.mz0 = 257;
            }
            else if IS_OPTION("l") {
                config.mx0 = 257;
                config.my0 = 257;
                config.mz0 = 513;
            }
            else if IS_OPTION("xl") {
                config.mx0 = 513;
                config.my0 = 513;
                config.mz0 = 513;
            }
            else {
                return E_INV_DSV;
            }
        }
        else if (IS_OPTION("-h") || IS_OPTION("--help")) {
            mode = MODE_HELP;
            return E_SUCCESS;
        }
        else {
            return E_UNKNOWN;
        }

        idx++;
    }

    return E_SUCCESS;

#undef IS_OPTION
}

inline void print_help() {
    char msg[512];
    sprintf(msg, h_msg, gargv[0]);
    cout << endl << msg << endl << endl;
}

inline void CheckError(int rc) {
    if (rc != E_SUCCESS) {
        cerr << endl << "Error: " << e_msg[rc] << endl;
        if (rc == E_NO_ARG)
            print_help();
        else
            cout << endl;
        exit(rc);
    }
}

int main(int argc, char ** argv) {
//    start_pes(0);
    CheckError(    SetPreference(argc, argv));
//    start_pes(0);
    timeval time_begin, time_end;
    double time_period;
    switch (mode) {
    case MODE_NORM:
        config.mimax = (config.ndx0 == 1) ?
                        config.mx0 : (config.mx0 / config.ndx0) + 3;
        config.mjmax = (config.ndy0 == 1) ?
                        config.my0 : (config.my0 / config.ndy0) + 3;
        config.mkmax = (config.ndz0 == 1) ?
                        config.mz0 : (config.mz0 / config.ndz0) + 3;
        bmtSetup(&argc, &argv);
    gettimeofday(&time_begin, NULL);
        bmtStart(blocks, threads);
        bmtClean();
        break;
    case MODE_HELP:
        print_help();
        break;
    }
    gettimeofday(&time_end, NULL);
    time_period = (time_end.tv_sec + time_end.tv_usec * 1e-6) - (time_begin.tv_sec + time_begin.tv_usec * 1e-6);
    printf("Time: %lf\n", time_period);
    return 0;
}

