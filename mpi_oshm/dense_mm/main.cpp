#include "main.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <mpi.h>
#ifdef _OSHM_MOD
#include <shmem.h>
#endif

#define HELP_S         "-h"
#define HELP_L         "--help"

#define MATRIX_SIZE_S  "-m"
#define MATRIX_SIZE_L  "--matrix"

#define SHOW_MATRIX_S  "-s"
#define SHOW_MATRIX_L  "--show"

#define NUM_BLOCKS_S   "-b"
#define NUM_BLOCKS_L   "--blocks"

#define NUM_THREADS_S  "-t"
#define NUM_THREADS_L  "--threads"
using namespace std;

char          * pDatasetFile = (char *) NULL; 
MatrixDataset * pMatrix;
pConfig       * pConf;
bool            showmat      = false;
int blocks;
int threads;

void setConfig(int argc, char ** argv){

    if (argc == 1){
        cout << "\n==== HELP ====\n-h or --help\tfor help\n"
            "-b or --blocks \tto choose number of blocks\n"
            "-t or --threads\tto choose number of threads\n"
            "-m or --matrix \tto choose different size binary file\n\n";
        exit(-1);
    }

    for (int i = 1; i < argc; i++){
        if ( !strcmp(argv[i], HELP_S) || !strcmp(argv[i], HELP_L) ) {
            cout << "\n==== HELP ====\n-h or --help\tfor help\n"
            "-b or --blocks \tto choose number of blocks\n"
            "-t or --threads\tto choose number of threads\n"
            "-m or --matrix \tto choose different size binary file\n\n";
        }
        else if ( !strcmp(argv[i], MATRIX_SIZE_S) || !strcmp(argv[i], MATRIX_SIZE_L) ) {
            pDatasetFile = argv[++i];
        }
        else if ( !strcmp(argv[i], NUM_BLOCKS_S) || !strcmp(argv[i], NUM_BLOCKS_L) ) {
            blocks = atoi(argv[++i]);
        }
        else if ( !strcmp(argv[i], NUM_THREADS_S) || !strcmp(argv[i], NUM_THREADS_L) ) {
            threads = atoi(argv[++i]);
        }
        else if ( !strcmp(argv[i], SHOW_MATRIX_S) || !strcmp(argv[i], SHOW_MATRIX_L) ) {
            showmat = true;
        }
    }
}


int main(int argc, char ** argv) {

    setConfig(argc, argv);

    timeval time_begin, time_end;
    double  time_period;
    int     numpes=1, peid=0;
// MPI definition
    int     rank, commSize;
    int     rankid;
    bool    result = false;

/**
* Determine if Openshmem or mpi version or serial version
**/

    pConf = new pConfig[1];
#ifdef _OSHM_MOD
    shmem_init();
    numpes = shmem_n_pes();
    peid   = shmem_my_pe();
    rankid = peid;
    pConf->mrank     = peid;
    pConf->mcommSize = numpes;
#endif

#ifdef _MPI_MOD
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    pConf->mrank     = rank;
    pConf->mcommSize = commSize;
    rankid           = rank;
#endif 
/**
* First PE read data from the datasetFile
**/
    if (rankid == 0) {
        if (pDatasetFile != NULL) {
            pMatrix = new MatrixDataset(pDatasetFile);
            if (showmat)
                pMatrix->showmatrix(result); 
        }
        printf("Kernel Config: blocks:%d, threads; %d\n", blocks, threads);
    }

    LibSetup(pMatrix, pConf);
#ifdef _OSHM_MOD
    shmem_barrier_all();
#endif
#ifdef _MPI_MOD
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    gettimeofday(&time_begin, NULL);
    LibEntry(showmat, blocks, threads, pMatrix, pConf);
    gettimeofday(&time_end, NULL);
    LibCleanUp();

    time_period = (time_end.tv_sec + time_end.tv_usec * 1e-6) - (time_begin.tv_sec + time_begin.tv_usec * 1e-6);
    printf("Time Main: %lf\n", time_period);

#ifdef _MPI_MOD
    MPI_Finalize();
#endif
#ifdef _OSHM_MOD
    shmem_finalize();
#endif
    return 0;    
}
