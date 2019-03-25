#include <unistd.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <time.h>
#include <fstream>
#include <string.h>

typedef void (* Shmem_fsumall) (float *target, 
    float *source, int nreduce, int PE_start,
    int logPE_stride, int PE_size, float *pWrk, 
    long *pSync);

typedef void (* Start_pe) (int npes);

typedef int (* Num_pe) (void);

typedef int (* My_pe) (void);

typedef void * (* Shmem_malloc) (size_t size);

typedef void (* Shmem_fget) (float *dest, const float *src, size_t nelems,int pe);

typedef int (* Shmem_ball) (void);

void * libHandler;
bool   isInitialized  = false;
Shmem_fsumall shfsuma = NULL;
Start_pe      shstpe  = NULL;
Num_pe        shnumpe = NULL;
My_pe         shmype  = NULL;
Shmem_malloc  shmal   = NULL;
Shmem_fget    shfget  = NULL;
Shmem_ball    shball  = NULL;

using namespace std;

extern "C" {

extern void start_pes (int npes) {

    timeval time0, time1;
    double  timestamp;

    if (!isInitialized) {
        libHandler = dlopen("/usr/local/osh/lib/libopenshmem.so",             RTLD_NOW);
        if (libHandler == NULL) {
            cout << "EXIT:Handler!" << endl;
            exit(-1);
        }
// start_pes(0)
        shstpe  = (Start_pe) dlsym(libHandler, "start_pes");
        if (shstpe == NULL) {
            cout << "EXIT:start_pes()." << endl;
            exit(-1);
        }
// _num_pes
        shnumpe = (Num_pe) dlsym(libHandler, "_num_pes");
        if (shnumpe == NULL) {
            cout << "EXIT:_num_pes()." << endl;
            exit(-1);
        }
// my_pe()
        shmype  = (My_pe) dlsym(libHandler, "_my_pe"); 
        if (shmype == NULL) {
            cout << "EXIT:_my_pe()." << endl;
            exit(-1);
        }
// shmalloc()
        shmal   = (Shmem_malloc) dlsym(libHandler, "shmalloc");
        if (shmal == NULL) {
            cout << "EXIT:shmalloc()." << endl;
            exit(-1);
        }
// shmem_float_get()
        shfget   = (Shmem_fget) dlsym(libHandler, "shmem_float_get");
        if (shfget == NULL) {
            cout << "EXIT:shmem_float_get()." << endl;
            exit(-1);
        }
// shmem_barrier_all()
        shball  = (Shmem_ball) dlsym(libHandler, 
            "shmem_barrier_all");
        if (shball == NULL) {
            cout << "EXIT:shmem_barrier_all()." << endl;
            exit(-1);
        }
// shmem_float_sum_to_all()
        shfsuma = (Shmem_fsumall) dlsym(libHandler, 
            "shmem_float_sum_to_all");
        if (shfsuma == NULL) {
            cout << "EXIT:shmem_float_sum_to_all()" << endl;
            exit(-1);
        }

        isInitialized = true;
    }

    fstream sfile;
    char filename[50];
    sprintf(filename, "start_pes.txt");
    sfile.open(filename, ios_base::out | ios_base::app);
    if (!sfile.is_open()) {
        cout << "Files Open Failed!" << endl;
        exit(-1);
    }

    char content[256], timech[50];
    memset(content, 0, sizeof(char) * 256);
    strcat(content, "\nInterception Begin......\n");
// Real function execution and time measurement
    gettimeofday(&time0, NULL);
    shstpe(npes);
    gettimeofday(&time1, NULL);

    timestamp = (time1.tv_sec*1e+6 + time1.tv_usec) -
        (time0.tv_sec*1e+6 + time0.tv_usec);
// Store the time stamp for this function
    strcat(content, "Interception Complete.\n");
    sprintf(timech, "\"start_pes()\" time is %lf usec\n", timestamp);
    strcat(content, timech);
    sfile.write(content, 256);
    sfile.close();

}

extern int _num_pes (void) {

    timeval time0, time1;
    double  timestamp;
    
    fstream sfile;
    char filename[50];
    sprintf(filename, "num_pes.txt");
    sfile.open(filename, ios_base::out | ios_base::app);
    if (!sfile.is_open()) {
        cout << "Files Open Failed!" << endl;
        exit(-1);
    }

char content[256], timech[50];
    memset(content, 0, sizeof(char) * 256);
    strcat(content, "\nInterception Begin......\n");

    gettimeofday(&time0, NULL);
    int npe = shnumpe();
    gettimeofday(&time1, NULL);

    timestamp = (time1.tv_sec*1e+6 + time1.tv_usec) -
        (time0.tv_sec*1e+6 + time0.tv_usec);

    strcat(content, "Interception Complete.\n");
    sprintf(timech, "\"_num_pes()\" time is %lf usec\n", timestamp);
    strcat(content, timech);
    sfile.write(content, 256);
    sfile.close();    
    
    return npe;
}
/***********************************************************/

extern int _my_pe (void) {

    timeval time0, time1;
    double  timestamp;

    fstream sfile;
    char filename[50];
    sprintf(filename, "my_pe.txt");
    sfile.open(filename, ios_base::out | ios_base::app);
    if (!sfile.is_open()) {
        cout << "Files Open Failed!" << endl;
        exit(-1);
    }

char content[256], timech[50];
    memset(content, 0, sizeof(char) * 256);
    strcat(content, "\nInterception Begin......\n");

    gettimeofday(&time0, NULL);
    int mpe = shmype();
    gettimeofday(&time1, NULL);

    timestamp = (time1.tv_sec*1e+6 + time1.tv_usec) -
        (time0.tv_sec*1e+6 + time0.tv_usec);

    strcat(content, "Interception Complete.\n");
    sprintf(timech, "\"_my_pe()\" time is %lf usec\n", timestamp);
    strcat(content, timech);
    sfile.write(content, 256);
    sfile.close();

    return mpe;
}

extern void *shmalloc (size_t size) {

    timeval time0, time1;
    double  timestamp;

    fstream sfile;
    char filename[50];
    sprintf(filename, "shmalloc.txt");
    sfile.open(filename, ios_base::out | ios_base::app);
    if (!sfile.is_open()) {
        cout << "Files Open Failed!" << endl;
        exit(-1);
    }

char content[256], timech[50];
    memset(content, 0, sizeof(char) * 256);
    strcat(content, "\nInterception Begin......\n");

    gettimeofday(&time0, NULL);
    void *ptr = shmal(size);
    gettimeofday(&time1, NULL);

    timestamp = (time1.tv_sec*1e+6 + time1.tv_usec) -
        (time0.tv_sec*1e+6 + time0.tv_usec);

    strcat(content, "Interception Complete.\n");
    sprintf(timech, "\"shmalloc()\" time is %lf usec\n", timestamp);
    strcat(content, timech);
    sfile.write(content, 256);
    sfile.close();

    return ptr;
}

extern void shmem_float_get (float *dest, const float *src, 
    size_t nelems,int pe) {

    timeval time0, time1;
    double  timestamp;

    fstream sfile;
    char filename[50];
    sprintf(filename, "shmem_float_get.txt");
    sfile.open(filename, ios_base::out | ios_base::app);
    if (!sfile.is_open()) {
        cout << "Files Open Failed!" << endl;
        exit(-1);
    }

char content[256], timech[50];
    memset(content, 0, sizeof(char) * 256);
    strcat(content, "\nInterception Begin......\n");

    gettimeofday(&time0, NULL);
    shfget(dest, src, nelems, pe);
    gettimeofday(&time1, NULL);

    timestamp = (time1.tv_sec*1e+6 + time1.tv_usec) -
        (time0.tv_sec*1e+6 + time0.tv_usec);

    strcat(content, "Interception Complete.\n");
    sprintf(timech, "\"shmem_float_get()\" size is: %lu time is %lf usec\n", nelems, timestamp);
    strcat(content, timech);
    sfile.write(content, 256);
    sfile.close();
}

extern void shmem_barrier_all (void) {

    timeval time0, time1;
    double  timestamp;

    fstream sfile;
    char filename[50];
    sprintf(filename, "shmem_barrier_all.txt");
    sfile.open(filename, ios_base::out | ios_base::app);
    if (!sfile.is_open()) {
        cout << "Files Open Failed!" << endl;
        exit(-1);
    }

char content[256], timech[50];
    memset(content, 0, sizeof(char) * 256);
    strcat(content, "\nInterception Begin......\n");

    gettimeofday(&time0, NULL);
    shball();
    gettimeofday(&time1, NULL);

    timestamp = (time1.tv_sec*1e+6 + time1.tv_usec) -
        (time0.tv_sec*1e+6 + time0.tv_usec);

    strcat(content, "Interception Complete.\n");
    sprintf(timech, "\"shmem_barrier_all\" time is %lf usec\n", timestamp);
    strcat(content, timech);
    sfile.write(content, 256);
    sfile.close();
}


/***********************************************************/
extern void shmem_float_sum_to_all(float *target, 

    float *source, int nreduce, int PE_start, 
    int logPE_stride, int PE_size, float *pWrk, 
    long *pSync) {

    timeval time0, time1;
    double  timestamp;

    fstream sfile;
    char filename[50];
    sprintf(filename, "shmem_f_sum_to_all.txt");
    sfile.open(filename, ios_base::out | ios_base::app);
    if (!sfile.is_open()) {
        cout << "Files Open Failed!" << endl;
        exit(-1);
    }

char content[256], timech[50];
    memset(content, 0, sizeof(char) * 256);
    strcat(content, "\nInterception Begin......\n");
// Real function execution and time measurement
    gettimeofday(&time0, NULL);
    shfsuma(target, source, nreduce, 
        PE_start, logPE_stride, PE_size, pWrk, pSync);
    gettimeofday(&time1, NULL);

    timestamp = (time1.tv_sec*1e+6 + time1.tv_usec) - 
        (time0.tv_sec*1e+6 + time0.tv_usec);
// Store the time stamp for this function
    strcat(content, "Interception Complete.\n");
    sprintf(timech, "\"shmem_float_sum_to_all()\" time is %lf usec\n", timestamp);
    strcat(content, timech);
    sfile.write(content, 256);
    sfile.close();

}

} //extern "C"
