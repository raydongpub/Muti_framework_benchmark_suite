/*
* File: nw_cpu.h
* Author: Da Li
* Email: da.li@mail.missouri.edu
* Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file has declaration of kernel functions on CPU.
*
*/

#ifndef __NEEDLE_CPU_H__
#define __NEEDLE_CPU_H__

#include "global.h"

void needleman_cpu_omp( char *sequence_set1, 
						char *sequence_set2, 
						unsigned int *pos1, 
						unsigned int *pos2,
						int *score_matrix, 
						unsigned int *pos_matrix,
						unsigned int max_pair_no, 
						short penalty);


float traceBack(int * score_matrix, char * seq1, char * seq2, 
				int size_row, int size_col, 
				int i, int j, int penalty);

#endif	// __NEEDLE_CPU_H__
