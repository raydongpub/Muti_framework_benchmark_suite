#include "test.decl.h"

CProxy_Main mainProxy;


void init_val(int num, int *dst, int *src) {
    for (int i=0; i<num; i++)
        dst[i] = src[i]+1;
} 
void update_val(int num, int*dst, int *src) {

}

class Main: public CBase_Main {
    CProxy_Grid mainPE, workPE;
public:
    Main(CkArgMsg* m) {
        if(m->argc>2) {
            CkAbort("Usage: no input\n");
        }

        mainProxy = thisProxy;

        mainPE = CProxy_Grid::ckNew(false, 10, 1);
        workPE = CProxy_Grid::ckNew(true, 2, 5);
        mainPE.SendInput(workPE);
        workPE.printout(2, CkCallback(CkReductionTarget(Grid, update_res), mainPE));
        //workPE.printout(2, CkCallback(CkReductionTarget(Main, done), mainPE));
        //mainPE.printmain(10);
    }
    void done() {
        CkExit();
    }
};

class Grid : public CBase_Grid {
    int * data;
    int number;
    Grid_SDAG_CODE
    public:
    Grid(bool accept, int work_num) : number(work_num){
        if (!accept) {
            data = new int[work_num];
            for (int i=0; i<work_num; i++) {
                data[i] = i+1;
            }
        } else {
            data = new int[work_num];
            for (int i=0; i<work_num; i++) {
                data[i] = 0;
            }
        }
    }
#if 1
    void update_res(CkReductionMsg *msg) {
        CkPrintf("My rank is: %d\n", thisIndex);
        CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
        int count = 0;
        while(current != NULL) {
            int *result = (int *) &current->data;
            data[count] = result[0];
            data[count+1] = result[1];
            count += 2;
            current = current->next();
        }
        printmain(10);
    }
#endif
    Grid(CkMigrateMessage*) {

    }

};

#include "test.def.h"
