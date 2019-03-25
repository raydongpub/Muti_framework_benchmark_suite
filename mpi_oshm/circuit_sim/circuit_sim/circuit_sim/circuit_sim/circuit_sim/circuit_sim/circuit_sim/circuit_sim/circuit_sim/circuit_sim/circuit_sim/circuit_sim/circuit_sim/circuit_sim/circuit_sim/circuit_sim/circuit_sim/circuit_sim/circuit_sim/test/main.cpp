#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main (void) {

    int * array = (int*) malloc(sizeof(int) * 9);
    for (int i=0; i<9; i++)
        array[i] = i+1;
    printf("array: \n");
    for (int i=0; i<9; i++)
        printf("%d ", array[i]);
    printf("\n");
    char * array2p = (char*)malloc(sizeof(int)*9);
    int ** array2d = (int **)malloc(sizeof(int*) * 3);
    for (int i=0; i<3; i++) {
        array2d[i] = reinterpret_cast<int *>(&array2p[i*3*sizeof(int)]);
    } 
    memcpy(array2d[0], array, 3*sizeof(int));
    memcpy(array2d[1], &array[3], 3*sizeof(int));
    memcpy(array2d[2], &array[6], 3*sizeof(int));
    printf("array2d: \n");
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            printf("%d ", array2d[i][j]);
    printf("\n");
    
    return 0;  
} 
