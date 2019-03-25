#include <stdio.h>
#include <stdlib.h>

int main() {
    unsigned char * mem_pool, *mem_begin, *mem_store;
    int mem_size;
    mem_size = sizeof(int) + 5 * sizeof(float);
    mem_pool = (unsigned char *) malloc(mem_size);
    mem_begin = mem_pool;
    mem_store = mem_pool;
/*
    float counter = 0.0;
    for (int i=0; i<5; i++) {
        *((float *) mem_pool) = counter;
        mem_pool += sizeof(float); 
        counter += 1.0;
    } 
*/
    int * index = reinterpret_cast<int *> (mem_pool);
    *index = 9;
    mem_pool += sizeof(int);
    float * pool;
    pool = reinterpret_cast<float *> (mem_pool);
    float counter = 0.0;
    for (int i=0; i<5; i++) {
        pool[i] = counter;
        counter += 1.0;
    } 
    unsigned char * rev_buf = mem_begin;
    int * rev = reinterpret_cast<int *>(rev_buf); 
    rev_buf += sizeof(int);
    float * rev_pool = reinterpret_cast<float *>(rev_buf);
    for (int i=0; i<5; i++) {
        printf("%f ", rev_pool[i]);
    }
    printf("\n");
    free(mem_begin); 
}
