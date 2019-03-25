/*
* File: nw_cpu.c 
 * Author: Da Li
 * Email: da.li@mail.missouri.edu
 * Organization: Networking and Parallel Systems Lab (http://nps.missouri.edu/)
*
* Description: This file defines kernel functions for alignment matrix calculation 
* and traceback of needleman-wunsch algorithm.
*
*/

#include <stdio.h>
#include "nw_cpu.h"

char blosum62_cpu[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  7, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};


short max_3( short a, short b, short c )
{
	short temp = a>b ? a: b; 
	return c>temp ? c: temp; 
}

void needleman_cpu(char *sequence_set1, 
				   char *sequence_set2, 
				   unsigned int *pos1, 
				   unsigned int *pos2,
				   int *score_matrix, 
				   unsigned int *pos_matrix,
				   unsigned int max_pair_no, 
				   short penalty)
{
	for (int i=0; i<max_pair_no; ++i){
			
		char *seq1 = sequence_set1+pos1[i];
		char *seq2 = sequence_set2+pos2[i];
		int seq1_len = pos1[i+1] - pos1[i];
		int seq2_len = pos2[i+1] - pos2[i];
		int *matrix = score_matrix + pos_matrix[i];
		matrix[0] = 0;		
		int dia, up, left;
		for(int j = 1; j <= seq1_len; ++j)
			matrix[j] = penalty * j;
		for(int j = 1; j <= seq2_len; ++j)
			matrix[j*(seq1_len+1)+0] = penalty * j;

		//fill the score matrix
		for(int k = 1; k <= seq2_len; ++k){           
			for(int j = 1; j <= seq1_len; ++j){						
				dia = matrix[(k-1)*(seq1_len+1)+j-1]+blosum62_cpu[ seq2[k-1] ][ seq1[j-1] ];
				up	= matrix[(k-1)*(seq1_len+1)+j] + penalty;
				left= matrix[k*(seq1_len+1)+j-1] + penalty;
				matrix[k*(seq1_len+1)+j] = max_3(left, dia, up);
			}
		}
	}
}

float traceBack(int * score_matrix, char * seq1, char * seq2,
                int size_row, int size_col,
                int i, int j, int penalty)
{
    int lenFinal = 0, gap = 0;
    int scoreFinal = score_matrix[i*size_row+j];
    int gapPenalty = 0;
    int flag = 0; // Indicate the gap
    int dia, up, left, curr;
	//printf("Size of row: %d\n", size_row);
    do{
        /*dia  = scoreMatrix[(i-1)*sizeRow+(j-1)] +
               substMat[(getIndex(seq2[i-1]) * substMatSize) + getIndex(seq1[j-1])];*/
        dia  = score_matrix[(i-1)*size_row+(j-1)] +
               blosum62_cpu[ seq2[i-1] ][ seq1[j-1] ];

        up   = score_matrix[(i-1)*size_row+j] + penalty;
        left = score_matrix[i*size_row+(j-1)] + penalty;
        curr = score_matrix[i*size_row+j];

        if ( curr == dia ){
            --i; --j;
            flag = 0; // close gap
        }
        else if ( curr==left ){
            --j; ++gap;
            gapPenalty += 2;
            if ( flag == 0 ){// gap opening
                gapPenalty += 10;
                flag = 1;
            }
        }
        else if ( curr==up ){
            --i; ++gap;
            gapPenalty += 2;
            if ( flag == 0 ){ // gap opening
                gapPenalty += 10;
                flag = 1;
            }
        }
        else {
            fprintf( stdout, "Error!");
            return 0.0;
        }
        ++lenFinal;
    }while( 0!=i && 0!=j);
    return (float)(scoreFinal-gapPenalty)/scoreFinal;
}
