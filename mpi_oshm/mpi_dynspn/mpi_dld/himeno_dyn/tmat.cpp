
#define IMAX 2
#define JMAX 3
#define KMAX 4

#if 1
#include "commonBMT.h"
#include <iostream>
#include <sys/time.h>

using namespace std;

int main(int argc, char ** argv) {

    Matrix * pa = new Matrix(4, IMAX, JMAX, KMAX);
    Matrix & a = *pa;
    PRECISION **** aaa = pa->GetPtr4D();

//    cout << &((*pa)(0,1,3)) << endl;

    struct timeval start, end;

    gettimeofday(&start, NULL);
    PRECISION cnt = 0;
    for (int i=0;i<IMAX;i++)
        for (int j=0;j<JMAX;j++)
            for (int k=0;k<KMAX;k++)
                aaa[0][i][j][k] = cnt++;
    gettimeofday(&end, NULL);
    PRECISION endd   = (PRECISION) (end.tv_sec * 1e+6)   + (PRECISION) (end.tv_usec);
    PRECISION startd = (PRECISION) (start.tv_sec * 1e+6) + (PRECISION) (start.tv_usec);
    cout << "Time: " << endd - startd << endl;

#if 0
    for (int i=0;i<IMAX;i++) {
        for (int j=0;j<JMAX;j++) {
            for (int k=0;k<KMAX;k++)
                cout << a(i, j, k) << " ";
            cout << endl;
        }
        cout << endl << endl << endl;
    }
#endif

    delete pa;

    return 0;
}
#else
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
    double a[IMAX][JMAX][KMAX];
int main(int argc, char ** argv) {
    double cnt = 0;
/*
    double *** a = (double ***) malloc(IMAX * sizeof(double **));
    for (int i=0;i<IMAX;i++) {
        a[i] = (double **) malloc(JMAX * sizeof(double *));
        for (int j=0;j<JMAX;j++)
            a[i][j] = (double *) malloc(KMAX * sizeof(double));
    }
*/
    struct timeval start, end;

    gettimeofday(&start, NULL);
    for (int i=0;i<IMAX;i++)
        for (int j=0;j<JMAX;j++)
            for (int k=0;k<KMAX;k++)
                a[i][j][k] = cnt++;
    gettimeofday(&end, NULL);
    double endd   = (double) (end.tv_sec * 1e+6)   + (double) (end.tv_usec);
    double startd = (double) (start.tv_sec * 1e+6) + (double) (start.tv_usec);
    printf("%f\n\n", endd - startd);
/*
    for (int i=0;i<IMAX;i++) {
        for (int j=0;j<JMAX;j++) {
            for (int k=0;k<KMAX;k++)
                printf("%d ", a[i][j][k]);
        }
    }
*/
//    printf("%p %p %p %p %p %p %p\n", a, a[0], a[1], &a[0][1], &a[0][2], &a[0][1][0], &a[0][1][3]);

    return 0;
}

#endif

