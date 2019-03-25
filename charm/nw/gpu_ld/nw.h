#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define MAX_SEQ_LEN 3000
#define MAX_SEQ_NUM 20000
#define MAX_STREAM 20

typedef struct {
        bool debug;
        int device;
        int kernel;
        int num_threads;
        int num_blocks;
        int num_streams;
        int num_pairs;
        int length;
        int penalty;
        int repeat;
        int dataset;
        int chare_num;
        int div;
        void pup(PUP::er &p) {
            p|debug;p|device;p|kernel;
            p|num_threads;p|num_blocks;p|num_streams;
            p|num_pairs;p|length;p|penalty;
            p|repeat;p|dataset;p|chare_num;
            p|div;

        }

} _CONF_;

/*extern char sequence_set1[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ];
extern char sequence_set2[MAX_STREAM][ MAX_SEQ_LEN * MAX_SEQ_NUM ];
extern unsigned int pos1[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int pos2[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int pos_matrix[MAX_STREAM][MAX_SEQ_NUM];
extern unsigned int dim_matrix[MAX_STREAM][MAX_SEQ_NUM];
extern char * d_sequence_set1[MAX_STREAM];
extern char * d_sequence_set2[MAX_STREAM];
extern unsigned int * d_pos1[MAX_STREAM];
extern unsigned int * d_pos2[MAX_STREAM];
extern int * d_score_matrix[MAX_STREAM];
extern unsigned int * d_pos_matrix[MAX_STREAM];
extern unsigned int * d_dim_matrix[MAX_STREAM];
extern int pair_num[MAX_STREAM];

double gettime();*/

#endif

