#include "himeno.decl.h"
#include "himeno.h"
#include "commonBMT.h"

#define wrapX(a)        (((a)+num_chare_x)%num_chare_x)
#define wrapY(a)        (((a)+num_chare_y)%num_chare_y)
#define wrapZ(a)        (((a)+num_chare_z)%num_chare_z)

/*readonly*/ int num_chare_x;
/*readonly*/ int num_chare_y; 
/*readonly*/ int num_chare_z;
int     gargc;
char ** gargv;
int     penum;
// Function
void print_help() {
    char msg[512];
    sprintf(msg, h_msg, gargv[0]);
    ckout << endl << msg << endl << endl;
}

void CHKERR(int line, cudaError_t ce)
{
    if (ce != cudaSuccess){
        ckout << "Error: line " << line << " "<< cudaGetErrorString(ce) << endl;
    }
}


int SetPreference(int argc, char ** argv, BMT_Config * config) {
#define IS_OPTION(str) (!strcmp(argv[idxx], str))
    int idxx = 1;
    gargc = argc; gargv = argv;


    if (argc < 2)
        return E_NO_ARG;

    while (idxx < argc) {
        if IS_OPTION("-pe") {
            if ((idxx + 3) >= argc)
                return E_INV_PE;
             int temp = atoi(argv[++idxx]);
             if (temp < 1)
                 return E_INV_PEV;
             config->ndx0 = temp;

             temp = atoi(argv[++idxx]);
             if (temp < 1)
                 return E_INV_PEV;
             config->ndy0 = temp;

             temp = atoi(argv[++idxx]);
             if (temp < 1)
                 return E_INV_PEV;
             config->ndz0 = temp;
        }
        else if (IS_OPTION("-ds") || IS_OPTION("--dataset-size")) {
            if ((idxx + 1) >= argc)
                return E_INV_DS;
            idxx++;
            if IS_OPTION("xs") {
                config->mx0 = 33;
                config->my0 = 33;
                config->mz0 = 65;
            }
            else if IS_OPTION("s") {
                config->mx0 = 65;
                config->my0 = 65;
                config->mz0 = 129;
            }
            else if IS_OPTION("m") {
                config->mx0 = 129;
                config->my0 = 129;
                config->mz0 = 257;
            }
            else if IS_OPTION("l") {
                config->mx0 = 257;
                config->my0 = 257;
                config->mz0 = 513;
            }
            else if IS_OPTION("xl") {
                config->mx0 = 513;
                config->my0 = 513;
                config->mz0 = 513;
            }
            else {
                return E_INV_DSV;
            }
        }
        else if (IS_OPTION("-h") || IS_OPTION("--help")) {
            print_help();
            return E_SUCCESS;
        }
        else if (IS_OPTION("-c") || IS_OPTION("--chare")) {
            penum = atoi(argv[++idxx]);
            return E_SUCCESS;
        }
        else {
            return E_UNKNOWN;
        }

        idxx++;
    }

    return E_SUCCESS;

#undef IS_OPTION
}


void CheckError(int rc) {
    if (rc != E_SUCCESS) {
        ckout << endl << "Error: " << e_msg[rc] << endl;
        if (rc == E_NO_ARG)
            print_help();
        else
            ckout << endl;
        exit(rc);
    }
}

// Main Proxy
CProxy_Main mainProxy;

class Main : public CBase_Main {
    double startTime;
    BMT_Config config;
    CProxy_Data dst;
    int idx;
    int iter;
public:
    Main(CkArgMsg *m) {
        CheckError(    SetPreference(m->argc, m->argv, &config));
        idx = 0;
        ckout <<"\t\tPE dimension: " << config.ndx0 << ":" << config.ndy0 << ":" << config.ndz0 << " and chare number: " << penum << endl;

// store the main proxy
        mainProxy = thisProxy;
        config.mimax = (config.ndx0 == 1) ?
            config.mx0 : (config.mx0 / config.ndx0) + 3;
        config.mjmax = (config.ndy0 == 1) ?
            config.my0 : (config.my0 / config.ndy0) + 3;
        config.mkmax = (config.ndz0 == 1) ?
            config.mz0 : (config.mz0 / config.ndz0) + 3;

        num_chare_x = config.ndx0;
        num_chare_y = config.ndy0;
        num_chare_z = config.ndz0;
        iter = 0;
// Create Chares
        dst = CProxy_Data::ckNew(config, penum, config.ndx0, config.ndy0, config.ndz0);
        dst.jacobi(CkCallback(CkReductionTarget(Main, collect), thisProxy));
        startTime = CkWallTimer();
    }
    void collect(float gosaval) {    
        iter++;
        ckout << "gosa: " << gosaval << endl;
        if (iter < 20) {
            if (iter % 2)
               dst.pauseForLB();
            else
               dst.jacobi(CkCallback(CkReductionTarget(Main, collect), thisProxy));
        }
        else {
            double endTime  = CkWallTimer();
            CkPrintf("Time: %lf\n", endTime-startTime);
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
        dst.jacobi(CkCallback(CkReductionTarget(Main, collect), thisProxy));
    }
};

class Data : public CBase_Data {
    Data_SDAG_CODE
public:
    BMT_Config conf;
    PRECISION **** a,    **** b,    **** c,   *** pt;
    PRECISION *** wrk1,  *** wrk2,  *** bnd;

/*    PRECISION * fa_d, * fb_d,* fc_d, * fp_d, * fwrk1_d, * fwrk2_d, 
              * fbnd_d, **** a_d, **** b_d, **** c_d, *** p_d,
              *** wrk1_d, *** wrk2_d, *** bnd_d, * gosa_d, 
              **** a_h, **** b_h, **** c_h, *** p_h, *** wrk1_h,  
              *** wrk2_h, *** bnd_h, * gosa_h;*/

    Matrix    * pa, * pb, * pc, * pp,   * pwrk1, * pwrk2, *pbnd;
    int       mx,   my,   mz,   imax,   jmax,    kmax,   it;
    PRECISION omega;
    PRECISION wgosa, gosa;
    size_t    memreq_3d;
    int       peid, penum, iter;
    int       remoteCount;
    bool copy, ctype;

    Data (BMT_Config config, int penum_) {
        usesAtSync = true;
        omega = 0.8;
        conf  = config;
        iter  = 0;
        copy  = true;
        memreq_3d = conf.mimax * conf.mjmax *
            conf.mkmax * sizeof(PRECISION);
        int dx,dy,dz;
        dx    = conf.ndx0;
        dy    = conf.ndy0;
        dz    = conf.ndz0;
        peid  = (thisIndex.x%dx)*dy*dz + (thisIndex.y%dy)*dz + thisIndex.z;
        penum = penum_;
        mx    = conf.mx0 - 1;
        my    = conf.my0 - 1;
        mz    = conf.mz0 - 1;

        bmtInitMax(); 
// Show work partition        
        ckout << "My PE[" << peid << "/"
            << penum << "]: ";
        ckout << "imax[" << imax << "] " << thisIndex.x;
        ckout << "jmax[" << jmax << "] " << thisIndex.y;
        ckout << "kmax[" << kmax << "] " << thisIndex.z;
        ckout << "it: " << it << endl;

        pa = new Matrix(4, conf.mimax, conf.mjmax, conf.mkmax);
        pb = new Matrix(3, conf.mimax, conf.mjmax, conf.mkmax);
        pc = new Matrix(3, conf.mimax, conf.mjmax, conf.mkmax);
        pp = new Matrix(conf.mimax, conf.mjmax, conf.mkmax);
        pwrk1 = new Matrix(conf.mimax, conf.mjmax, conf.mkmax);
        pwrk2 = new Matrix(conf.mimax, conf.mjmax, conf.mkmax);
        pbnd  = new Matrix(conf.mimax, conf.mjmax, conf.mkmax);

        bmtInitMt(
            *pa,   *pb,    *pc,
            *pp,   *pwrk1, *pwrk2,
            *pbnd,  mx,     it,
             conf.mimax,  conf.mjmax, conf.mkmax,
             imax,   jmax,  kmax);
// CudaInitialize

        a    = pa->GetPtr4D();
        b    = pb->GetPtr4D();
        c    = pc->GetPtr4D();
        pt   = pp->GetPtr3D();
        wrk1 = pwrk1->GetPtr3D();
        wrk2 = pwrk2->GetPtr3D();
        bnd  = pbnd->GetPtr3D();


    }
    void pup(PUP::er &p) {
        ckout << "\t\t###### Migration !!!" << endl;
        CBase_Data::pup(p);
        __sdag_pup(p);
        p|mx; p|my; p|mz; p|imax; p|jmax; p|kmax; p|it;
        p|conf; p|peid; p|penum; p|omega; p|wgosa; p|gosa; p|iter; p|remoteCount; p|copy;
        p|memreq_3d; p|ctype; 
        if (p.isUnpacking()) {
            pa = new Matrix(4, conf.mimax, conf.mjmax, conf.mkmax);
            pb = new Matrix(3, conf.mimax, conf.mjmax, conf.mkmax);
            pc = new Matrix(3, conf.mimax, conf.mjmax, conf.mkmax);
            pp = new Matrix(conf.mimax, conf.mjmax, conf.mkmax);
            pwrk1 = new Matrix(conf.mimax, conf.mjmax, conf.mkmax);
            pwrk2 = new Matrix(conf.mimax, conf.mjmax, conf.mkmax);
            pbnd  = new Matrix(conf.mimax, conf.mjmax, conf.mkmax);
            a    = pa->GetPtr4D();
            b    = pb->GetPtr4D();
            c    = pc->GetPtr4D();
            pt   = pp->GetPtr3D();
            wrk1 = pwrk1->GetPtr3D();
            wrk2 = pwrk2->GetPtr3D();
            bnd  = pbnd->GetPtr3D();
        }
        PUParray(p, pa->mpVal, pa->count);
        PUParray(p, pb->mpVal, pb->count);
        PUParray(p, pc->mpVal, pc->count);
        PUParray(p, pp->mpVal, pp->count);
        PUParray(p, pwrk1->mpVal, pwrk1->count);
        PUParray(p, pwrk2->mpVal, pwrk2->count);
        PUParray(p, pbnd->mpVal, pbnd->count);
//        PUParray(p, localBuf, localCnt);
    }

    Data (CkMigrateMessage* m) { }
    void print_info () {
        ckout << "My PE[" << peid << "/"
            << penum << "]" << endl;

    }
    void bmtInitMax() {
//Work division and assignment for PEs

        int * mx1, * my1, * mz1;
        int * mx2, * my2, * mz2;
        int   tmp;

        mx1 = new int [conf.mx0 + 1];
        my1 = new int [conf.my0 + 1];
        mz1 = new int [conf.mz0 + 1];

        mx2 = new int [conf.mx0 + 1];
        my2 = new int [conf.my0 + 1];
        mz2 = new int [conf.mz0 + 1];

        tmp = mx / conf.ndx0;
        mx1[0] = 0;
        for (int i=1;i<=conf.ndx0;i++) {
            if (i <= mx % conf.ndx0)
                mx1[i] = mx1[i - 1] + tmp + 1;
            else
                mx1[i] = mx1[i - 1] + tmp;
        }

        tmp = my / conf.ndy0;
        my1[0] = 0;
        for (int i=1;i<=conf.ndy0;i++) {
            if (i <= my % conf.ndy0)
                my1[i] = my1[i - 1] + tmp + 1;
            else
                my1[i] = my1[i - 1] + tmp;
        }

        tmp = mz / conf.ndz0;
        mz1[0] = 0;
        for (int i=1;i<=conf.ndz0;i++) {
            if (i <= mz % conf.ndz0)
                mz1[i] = mz1[i - 1] + tmp + 1;
            else
                mz1[i] = mz1[i - 1] + tmp;
        }

    //************************************************************************

        for (int i=0;i<conf.ndx0;i++) {
            mx2[i] = mx1[i+1] - mx1[i];
            if (i != 0)
                mx2[i] = mx2[i] + 1;
            if (i != conf.ndx0-1)
                mx2[i] = mx2[i] + 1;
        }
        for (int i=0;i<conf.ndy0;i++) {
            my2[i] = my1[i+1] - my1[i];
            if (i != 0)
                my2[i] = my2[i] + 1;
            if (i != conf.ndy0-1)
                my2[i] = my2[i] + 1;
        }

        for (int i=0;i<conf.ndz0;i++) {
            mz2[i] = mz1[i+1] - mz1[i];
            if (i != 0)
                mz2[i] = mz2[i] + 1;
            if( i != conf.ndz0-1)
                mz2[i] = mz2[i] + 1;
        }

    //************************************************************************

        imax = mx2[thisIndex.x];
        jmax = my2[thisIndex.y];
        kmax = mz2[thisIndex.z];

        if (thisIndex.x == 0)
            it = mx1[thisIndex.x];
        else
            it = mx1[thisIndex.x] - 1;


        delete [] mz2;
        delete [] my2;
        delete [] mx2;

        delete [] mz1;
        delete [] my1;
        delete [] mx1;


    }
#if 0    
    void begin_iteration() {
    // Copy different faces into messages
        PRECISION *leftGhost   =  new PRECISION[jmax*kmax];
        PRECISION *rightGhost  =  new PRECISION[jmax*kmax];
        PRECISION *topGhost    =  new PRECISION[imax*kmax];
        PRECISION *bottomGhost =  new PRECISION[imax*kmax];
        PRECISION *frontGhost  =  new PRECISION[imax*jmax];
        PRECISION *backGhost   =  new PRECISION[imax*jmax];

        for(int k=0; k<kmax; ++k)
          for(int j=0; j<jmax; ++j) {
            leftGhost[k*jmax+j] = pt[1][j][k];
            rightGhost[k*jmax+j] = pt[imax-2][j][k];
        }

        for (int k=0; k<kmax; ++k)
            for(int i=0; i<imax; ++i) {
            topGhost[k*imax+i] = pt[i][1][k];
            bottomGhost[k*imax+i] = pt[i][jmax-2][k];
        }

        for (int j=0; j<jmax; ++j)
            for(int i=0; i<imax; ++i) {
            frontGhost[j*imax+i] = pt[i][j][1];
            backGhost[j*imax+i] = pt[i][j][kmax-2];
        }

        int x = thisIndex.x, y = thisIndex.y, z = thisIndex.z;

        thisProxy(wrapX(x-1),y,z).updateGhosts(RIGHT, jmax, kmax, rightGhost);
        thisProxy(wrapX(x+1),y,z).updateGhosts(LEFT, jmax, kmax, leftGhost);
        thisProxy(x,wrapY(y-1),z).updateGhosts(TOP, imax, kmax, topGhost);
        thisProxy(x,wrapY(y+1),z).updateGhosts(BOTTOM, imax, kmax, bottomGhost);
        thisProxy(x,y,wrapZ(z-1)).updateGhosts(BACK, imax, jmax, backGhost);
        thisProxy(x,y,wrapZ(z+1)).updateGhosts(FRONT,  imax, jmax, frontGhost);

        delete [] leftGhost;
        delete [] rightGhost;
        delete [] bottomGhost;
        delete [] topGhost;
        delete [] frontGhost;
        delete [] backGhost;
    }
#endif
    void updateBoundary(int dir, int height, int width, PRECISION * gh) {
        switch(dir) {
            case LEFT:
                for (int k=0; k<width; ++k)
                    for (int j=0; j<height; ++j) {
                        pt[0][j][k] = gh[k*height+j];
                } break;
            case RIGHT:
                for (int k=0; k<width; ++k)
                    for (int j=0; j<height; ++j) {
                        pt[imax-1][j][k] = gh[k*height+j];
                } break;
            case BOTTOM:
                for (int k=0; k<width; ++k)
                    for (int i=0; i<height; ++i) {
                        pt[i][0][k] = gh[k*height+i];
                } break;
            case TOP:
                for (int k=0; k<width; ++k)
                    for (int i=0; i<height; ++i) {
                        pt[i][jmax-1][k] = gh[k*height+i];
                } break;
            case FRONT:
                for (int j=0; j<width; ++j)
                    for (int i=0; i<height; ++i) {
                        pt[i][j][0] = gh[j*height+i];
                } break;
            case BACK:
                for (int j=0; j<width; ++j)
                    for (int i=0; i<height; ++i) {
                        pt[i][j][kmax-1] = gh[j*height+i];
                } break;
            default:
                CkAbort("ERROR\n");
        }
    }


    void bmtstart() {

       // if (iter < 10) { 
            cudaError_t ce = bmtInitDeviceMemory(
                             pa,   pb,    pc,
                             pp,   pwrk1, pwrk2,
                             pbnd, CkMyPe(), conf, copy
/*              &fa_d, &fb_d, &fc_d, &fp_d, &fwrk1_d, &fwrk2_d,
              &fbnd_d, &a_d, &b_d, &c_d, &p_d,
              &wrk1_d, &wrk2_d, &bnd_d, &gosa_d,
              &a_h, &b_h, &c_h, &p_h, &wrk1_h,
              &wrk2_h, &bnd_h, &gosa_h*/);

            if (ce != cudaSuccess)
                ckout << "Error: " << cudaGetErrorString(ce) << endl;

            ce = bmtCudaJacobi(&wgosa, pp,
                imax, jmax, kmax, conf, CkMyPe()
/*                fa_d, fb_d, fc_d, fp_d, fwrk1_d, fwrk2_d,
                fbnd_d, a_d, b_d, c_d, p_d,
                wrk1_d, wrk2_d, bnd_d, gosa_d,
                a_h, b_h, c_h, p_h, wrk1_h,
                wrk2_h, bnd_h, gosa_h*/);
            ckout << "\t\t********iter: " << iter << endl;
            if (ce != cudaSuccess) {
                ckout << "Error: " << cudaGetErrorString(ce) << endl;
            }
       // } else {
        //    mainProxy.done();
        //}
         
//        ckout << "My PE[" << CkMyPe() << "/"
//            << CkNumPes() << "] Compute complete" << endl;

    }
    void pauseForLB() {
        CkPrintf("Element %d pause for LB on PE %d\n", thisIndex, CkMyPe());
        AtSync();
    }
    void ResumeFromSync() {
        CkCallback cbld(CkReductionTarget(Main, resumeIter), mainProxy);
        contribute(sizeof(bool), &ctype, CkReduction::set, cbld);
        //CkCallback cbld(CkReductionTarget(Main, collect), mainProxy);
        //contribute(sizeof(PRECISION), &wgosa, CkReduction::sum_float, cbld); 
    }


};
#include "himeno.def.h"
