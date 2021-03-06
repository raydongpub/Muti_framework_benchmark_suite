#include "matmul.decl.h"
#include "rand48_replacement.h"
#include "matmul.h"
#include <math.h>

CProxy_Main mainProxy;

extern void cudaMatMul(int heg, int wid, int wh, int off, int subnum, PRECISION *matA, PRECISION *matB, PRECISION *matC, int peid_, int pid);

void cudaIniVar(int heg, int wid, int wh, int off, int subnum, PRECISION *matA, PRECISION *matB, PRECISION *matC, int id) {
    int peid   = id;
    int offset = peid * off;
    cudaMatMul(heg, wid, wh, offset, subnum, matA, matB, matC, peid, CkMyPe());

}

void display_mat(PRECISION *mat, int heg, int wid, int woid) {
    int total;
    cout << endl << "\t\tThe matrix display:" << endl;
    switch (woid) {
        case 0:
            total = heg * heg;
            for (int i=0; i<total; i++) {
                if (i%heg == 0) {
                    cout << "\t\t" << endl;
                }
                cout << mat[i] << " "; 
            }
            cout << endl; break;
        case 1:
            total = heg * wid;
            for (int i=0; i<total; i++) {
                if (i%heg == 0) {
                    cout << "\t\t" << endl;
                }
                cout << mat[i] << " ";
            }
            cout << endl; break;
        case 2:
            total = wid * wid;
            for (int i=0; i<total; i++) {
                 if (i%wid == 0) {
                 cout << "\t\t" << endl;
            }
            cout << mat[i] << " ";
            }
            cout << endl; break;
    }

}

class Main : public CBase_Main {
    double      startTime, midTime;
    PRECISION   *dataA, *dataB;
    int         height, width, workw, workh, peset;
    int         iter, parnum, subnum;
    int         worksplit, workheg, workwid;
    CProxy_Data c;
    char        *pDatasetFile;

public:
    Main(CkArgMsg* m) {
        if (m->argc > 6) {
            height       = atoi(m->argv[1]);
            width        = atoi(m->argv[2]);
            workh        = atoi(m->argv[3]); 
            workw        = atoi(m->argv[4]); 
            peset        = atoi(m->argv[5]); 
            pDatasetFile = m->argv[6];
        } else {
            CkAbort("Usage: matmul dimension");
        }
        int total_num = height * width;
        int penum = CkNumPes();


        if (height % workh != 0) {
            CkAbort("Warning: Matrix cannot be divided equaly by Height Partition");
        } else {
            workheg = height/workh;
        }
        if (width % workw != 0) {
            CkAbort("Warning: Matrix cannot be divided equaly by Width Partition");
        } else {
            workwid = width/workw;
        }
        if (workwid % peset != 0) {
            CkAbort("Warning: Matrix cannot be divided equaly by PEs");
        } else {
            worksplit = workwid/peset;
        }
        mainProxy = thisProxy; 
//Chare array
        iter = 0; 
        subnum = workheg * worksplit; 
        parnum = workh * workw;
// Get Data from .nn file
        MatrixDataset * pMatrix = new MatrixDataset(pDatasetFile);
        ckout << "The total number of PEs: " << penum << " and Partion is: " << parnum << endl;
        ckout << "The workportion is: " << subnum << endl;        
                  
        dataA = pMatrix->mpMatrix.elements;
        dataB = dataA + total_num; 
//        ckout <<"Get through!" << endl;

//        display_mat(dataA, height, width, 0);
//        display_mat(dataB, height, width, 0);
/*        a = CProxy_Data::ckNew(0, dataA, height, 
            width, worksplit, penum);
        b = CProxy_Data::ckNew(1, dataB, height, 
            width, worksplit, penum);*/
// Matrix B turning
        for (int i=0; i<width; i++) {
            for (int j=(i*width+i%width+1); j<(i+1)*width; j++) {
                int tmp = dataB[j];
                dataB[j] = dataB[(j%width)*width + (j/width)];
                dataB[(j%width)*width + (j/width)] = tmp;
            }
        }
        startTime = CkWallTimer();
//        display_mat(dataB, height, width, 0);
//        CkAbort("\t\ttest\n");
 //       for (int i=0; i<parnum; i++) {
/****** Work ******/
        int offset0  = (iter/workw) * workheg * width;
        int offset1  = (iter%workw) * workwid * height;
/******************/
        c = CProxy_Data::ckNew((dataA+offset0), (dataB+offset1), 
            workheg, workwid, height, width, peset, worksplit, parnum, peset);
//        }
//        delete pMatrix;

//        display_mat(dataA, height, width, 0);
//        display_mat(dataB, height, width, 0);

//        a.matInput(c, true);
//        b.matInput(c, false);
//        for (int i=0; i<parnum; i++) 
//        c.matRun(CkCallback(CkReductionTarget(Main, done), thisProxy));
//        midTime = CkWallTimer();
        c.matRun(CkCallback(CkReductionTarget(Main, done), thisProxy));
        //c.matRun();
 
    }
    
    void done(CkReductionMsg *msg) {
        CkPrintf("Resume iteration at step %d\n", iter);
        CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
        bool *result;
        if (current != NULL) {
            result = (bool *) &current->data;
        }
        iter++;
        if (iter < parnum) {
            if (iter % 2)
                c.pauseForLB();
            else
                c.matRun(CkCallback(CkReductionTarget(Main, done), thisProxy));
        }
        else {
            double endTime = CkWallTimer();
            CkPrintf("Time: %f\n", endTime, endTime - startTime);
            CkExit();
        }
   }
   void resumeIter(CkReductionMsg *msg) {
       CkPrintf("Resume iteration at step %d\n", iter);
       CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
       bool *result;
       if (current != NULL) {
           result = (bool *) &current->data;
       }
       c.matRun(CkCallback(CkReductionTarget(Main, done), thisProxy));
   }
};


class Data : public CBase_Data {
    int height, width, wheg, wwid, workdata, inid, petotal, iternum, iter, pid;
    int sizeA, sizeB, sizeRes, renum;
    double timestep;
    bool ctype;
    PRECISION *matA, *matB, *matC;
    Data_SDAG_CODE
public:
    Data(PRECISION *inidataA, PRECISION *inidataB, int wheight_, int wwidth_, int height_, int width_, int peset, int workportion, int iternum_): 
        height(height_), width(width_), wheg(wheight_), wwid(wwidth_), workdata(workportion) {
// Load balance switch on
//        ckout <<"flag0" << endl;
        usesAtSync = true;

//        ckout << "\t\tCID:" << thisIndex << "InitiaDataBegin: " << CkWallTimer() << endl; 
        int totalnum = height * width;
        sizeA        = wheg * width;
        sizeB        = workdata * height;
        renum        = workdata * wheg;
        iternum      = iternum_;
        iter         = 0;
        int workid   = thisIndex; 
        int penum    = CkNumPes();
        inid         = thisIndex;
        pid          = CkMyPe();
        petotal      = peset;
        sizeRes      = renum;
        matA = new PRECISION[sizeA];
        matB = new PRECISION[sizeB];
        matC = new PRECISION[renum];
        int offsetB = workid * workdata * height;
        for (int i=0; i<sizeA; i++) { 
            matA[i] = inidataA[i];    
        }
        for (int i=0; i<sizeB; i++) {
            matB[i] = inidataB[i + offsetB];
        }
        for (int i=0; i<renum; i++) {
             matC[i] = 0.0;
        }

        char hostname[128];
        gethostname(hostname, 128);
        //ckout << "\t\tCID:" << pid << "Hostname: : " << hostname << endl; 
    }        
    
    void pup(PUP::er &p) {
        CBase_Data::pup(p);
        __sdag_pup(p);
        p|height; p|width; p|wheg; p|wwid; p|workdata; p|renum; p|iternum; p|iter;
        p|sizeA; p|sizeB; p|sizeRes; p|petotal; p|inid; p|pid;
        p|ctype;
        if (p.isUnpacking()) {
            matA = new PRECISION[sizeA];
            matB = new PRECISION[sizeB];
            matC = new PRECISION[sizeRes];
        }
        p(matA, sizeA);
        p(matB, sizeB);
        p(matC, sizeRes);
        
    }

    Data (CkMigrateMessage*) {}
    void pauseForLB() {
        CkPrintf("Element %d pause for LB on PE %d\n", thisIndex, CkMyPe());
        AtSync();
    }
    void ResumeFromSync() {
        //workPE.pgmrun(CkCallback(CkReductionTarget(Grid, update_post), mainPE));
        //CkCallback cbld(CkIndex_Main::resumeIter(), mainProxy);
        CkCallback cbld(CkReductionTarget(Main, resumeIter), mainProxy);
        contribute(sizeof(bool), &ctype, CkReduction::set, cbld);
    }
};
#include "matmul.def.h"
