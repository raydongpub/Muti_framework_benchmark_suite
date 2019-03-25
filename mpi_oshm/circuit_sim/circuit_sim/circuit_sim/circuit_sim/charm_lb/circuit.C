#include "circuit.decl.h"
#include "circuitcuda.h"
#include "stdlib.h"
#include <vector>
#include <unistd.h>
#include <sys/time.h>

#define DLB

CProxy_Main mainProxy;
int num_pieces;
int max_pe;
int number_loops = 1;
int nodes_per_piece   = 2;
int wires_per_piece   = 4;
int pct_wire_in_piece = 95;
int random_seed       = 0;
int num_blocks        = 32;
int num_threads       = 256;

/**
 * @ data type definition
 */
//#define PRECISION float
//#define WIRE_SEGMENTS 10
//#define STEPS         10000
//#define DELTAT        1e-6
//
//#define INDEX_TYPE    unsigned
//#define INDEX_DIM     1
//
//#define D_NODE        0x0000
//#define D_WIRE        0x0001
//#define D_RAND
#define RAND_NUM 0.11
/**
 * @ Functions
 */
void init_val(int num, unsigned char *dst, unsigned char *src) {
    memcpy(dst, src, num);
} 
void init_post(cct * c_pc, PRECISION * src, int peid, int nodes_per_pc, int num_pc) {
    for (int i=0; i<num_pc; i++) {
        int disp = peid * num_pc * nodes_per_pc + i * nodes_per_pc;
        memcpy(c_pc->nodep[i].charge, &src[disp], nodes_per_pc * sizeof(PRECISION));
    }
}
void init_post_shr(cct * c_pc, PRECISION * src, int peid, int wires_per_pc, int num_pc) {
    for (int i=0; i<num_pc; i++) {
        int disp = peid * num_pc * wires_per_pc + i * wires_per_pc;
        memcpy(c_pc->wirep[i].shr_voltage, &src[disp], wires_per_pc * sizeof(PRECISION));
    }
}
void update_val(int num, int*dst, int *src) {

}
int random_element(int vec_size)
{
#ifndef D_RAND
  int index = int(drand48() * vec_size);
#else
  int index = int(RAND_NUM * vec_size);
#endif
  return index;
}
//cuda functions
extern void cudaRun(cct * cct_pc, unsigned char * buf, int nodes_per_pc, int wires_per_pc, int pieces_per_pe, int peid, int gpu_block, int gpu_thread);
extern void cudaPost(cct * cct_pc, unsigned char * buf, int nodes_per_pc, int wires_per_pc, int pieces_per_pe, int peid, int gpu_block, int gpu_thread);
void cudaInit(bool first, cct * cct_pc, unsigned char * buf, int nodes_per_pc, int wires_per_pc, int pieces_per_pe, int peid, int gpu_block, int gpu_thread) {
    if (first)
        cudaRun(cct_pc, buf, nodes_per_pc, wires_per_pc, pieces_per_pe, peid, gpu_block, gpu_thread);
    else
        cudaPost(cct_pc, buf, nodes_per_pc, wires_per_pc, pieces_per_pe, peid, gpu_block, gpu_thread);
}

class Main: public CBase_Main {
    CProxy_Grid mainPE, workPE;
    int loop_count;
public:
    Main(CkArgMsg* m) {
        if (m->argc<5) {
            CkPrintf("[num_pieces:%d], [num_nodes:%d], [num_wires:%d], [num_loops:%d], [max_pe:%d]\n", num_pieces, nodes_per_piece, wires_per_piece, number_loops, max_pe);
            CkAbort("[num_pieces], [num_nodes], [num_wires], [num_loops], [max_pe]\n");
        } 
        if (m->argc>=6) {
            num_pieces      = atoi(m->argv[1]);
            nodes_per_piece = atoi(m->argv[2]);
            wires_per_piece = atoi(m->argv[3]);
            number_loops    = atoi(m->argv[4]);
            max_pe          = atoi(m->argv[5]);
            CkPrintf("[num_pieces:%d], [num_nodes:%d], [num_wires:%d], [num_loops:%d], [max_pe:%d]\n", num_pieces, nodes_per_piece, wires_per_piece, number_loops, max_pe);
        }
        //work partition
        int pieces_per_pe = num_pieces / max_pe;

        mainProxy = thisProxy;
        loop_count = 0;

        mainPE = CProxy_Grid::ckNew(false, num_pieces, 1);
        workPE = CProxy_Grid::ckNew(true, pieces_per_pe, max_pe);

        mainPE.SendInput(workPE);
        workPE.pgmrun(CkCallback(CkReductionTarget(Grid, update_post), mainPE));
    }
    void post_run() {
        mainPE.SendPost(workPE);
        workPE.postrun(CkCallback(CkReductionTarget(Grid, update_res), mainPE));
    }
    void done() {
        loop_count++;
        CkPrintf("Ieration at step %d\n", loop_count);
        if (loop_count < number_loops) {
#ifdef DLB
            if (loop_count % 2) {
                //mainPE.pauseForLB();
                workPE.pauseForLB();
                //return;
            }
            else {
                mainPE.SendLoop(workPE);
                workPE.pgmrunloop(CkCallback(CkReductionTarget(Grid, update_post), mainPE));
            }
#else
            mainPE.SendLoop(workPE);
            workPE.pgmrunloop(CkCallback(CkReductionTarget(Grid, update_post), mainPE));
#endif
        }
        else {
            CkPrintf("End of program!\n");
            CkExit();
        }
    }
    void resumeIter(CkReductionMsg *msg) {
        CkPrintf("Resume iteration at step %d\n", loop_count);
        CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
        bool *result;
        if (current != NULL) {
            result = (bool *) &current->data;
        }
        if (* result == true) {
            CkPrintf("WorkPE %d\n", loop_count);
            workPE.pgmrunloop(CkCallback(CkReductionTarget(Grid, update_post), mainPE));
        }
        else {
            CkPrintf("MainPE %d\n", loop_count);
            mainPE.SendLoop(workPE);
        }
        
#if 0
        mainPE.SendLoop(workPE);
        workPE.pgmrunloop(CkCallback(CkReductionTarget(Grid, update_post), mainPE));
#else

        //CkPrintf("End of program!\n");
        //CkExit();
#endif
    }
};

class Grid : public CBase_Grid {
    cct  * circuit_pc;
    PRECISION * post_charge, * post_shr_voltage;
    unsigned char * mem_pool, * mem_begin, * transfer_buf, * result_buf;
    int mem_pc_size, mem_size, transfer_size, result_size; 
    int num_pieces;
    int post_num, post_shr_num;
    bool ctype;
    Grid_SDAG_CODE
    public:
    Grid(bool accept, int num_pieces_) : num_pieces(num_pieces_){

        CkPrintf("Rank: %d, [num_pieces:%d], [num_nodes:%d], [num_wires:%d], [num_loops:%d], [max_pe:%d]\n", thisIndex, num_pieces, nodes_per_piece, wires_per_piece, number_loops, max_pe);
//        circuit_pc->nodep  = new node[num_pieces];
//        circuit_pc->wirep = new wire[num_pieces];
        mem_pc_size = nodes_per_piece * (sizeof(PRECISION) * 4 + sizeof(int) * 2)
                 + wires_per_piece * (sizeof(PRECISION) * (2 * WIRE_SEGMENTS + 4) + sizeof(int) * 4);
        mem_size = num_pieces * mem_pc_size;
        //mem_pool = (unsigned char *)malloc(mem_size);
        //mem_begin = mem_pool;
        circuit_pc = new cct(num_pieces, wires_per_piece, nodes_per_piece, mem_pc_size, mem_size);
        ctype = accept;
#if 1
        if (!accept) {
            // set random seed
            srand48(random_seed);
            std::vector<int> shared_nodes_piece(num_pieces, 0);
            // node initialization
#ifdef _DEBUG
                CkPrintf("=== List nodes in piece===\n");
#endif
            for (int n = 0; n < num_pieces; n++) {
                // initialize node_per_piece
                for (int i = 0; i < nodes_per_piece; i++) {
                    // initialize node parameter
#ifndef D_RAND
                    circuit_pc->nodep[n].capacitance[i] = drand48() + 1.f;
                    circuit_pc->nodep[n].leakage[i]     = 0.1f * drand48();
                    circuit_pc->nodep[n].charge[i]      = 0.f;
                    circuit_pc->nodep[n].voltage[i]     = 2*drand48() - 1.f;
#else
                    circuit_pc->nodep[n].capacitance[i] = RAND_NUM + 1.f;
                    circuit_pc->nodep[n].leakage[i]     = 0.1f * RAND_NUM;
                    circuit_pc->nodep[n].charge[i]      = 0.f;
                    circuit_pc->nodep[n].voltage[i]     = 2*RAND_NUM - 1.f;
#endif
                    // node_attr (0:private, 1:shared)
                    circuit_pc->nodep[n].shr_pc[i]      = 0;
                    circuit_pc->nodep[n].node_attr[i]   = 0;
#ifdef _DEBUG
                    CkPrintf("\tvoltage: %f, charge: %f\n", circuit_pc->nodep[n].voltage[i], circuit_pc->nodep[n].charge[i]);
#endif
                }//for
            }//for

            // wire initialization
            for (int n = 0; n < num_pieces; n++) {
                for (int j=0; j<wires_per_piece; j++) {
                    circuit_pc->wirep[n].shr_voltage[j] = 0.f;
                    circuit_pc->wirep[n].shr_charge[j]  = 0.f;
                    circuit_pc->wirep[n].shr_pc[j]      = 0;
                    circuit_pc->wirep[n].wire_attr[j]   = 0;
                }
                // initialize wire parameter
                for (int i = 0; i < wires_per_piece; i++) {
                    // init currents
                    for (int j = 0; j < WIRE_SEGMENTS; j++)
                        circuit_pc->wirep[n].currents[i][j] = 0.f;
                    // init voltage
                    for (int j = 0; j < WIRE_SEGMENTS-1; j++)
                        circuit_pc->wirep[n].voltages[i][j] = 0.f;
#ifndef D_RAND
                    // init resistance
                    circuit_pc->wirep[n].resistance[i]  = drand48() * 10.0 + 1.0;
                    // Keep inductance on the order of 1e-3 * dt to avoid resonance problems
                    circuit_pc->wirep[n].inductance[i]  = (drand48() + 0.1) * DELTAT * 1e-3;
                    circuit_pc->wirep[n].capacitance[i] = drand48() * 0.1;
#else
                    // init resistance
                    circuit_pc->wirep[n].resistance[i]  = RAND_NUM * 10.0 + 1.0;
                    // Keep inductance on the order of 1e-3 * dt to avoid resonance problems
                    circuit_pc->wirep[n].inductance[i]  = (RAND_NUM + 0.1) * DELTAT * 1e-3;
                    circuit_pc->wirep[n].capacitance[i] = RAND_NUM * 0.1;

#endif
                    // UNC init connection
#ifndef D_RAND
                    circuit_pc->wirep[n].in_ptr[i] = random_element(nodes_per_piece);
                    if ((100 * drand48()) < pct_wire_in_piece) {
                        circuit_pc->wirep[n].out_ptr[i] = random_element(nodes_per_piece);
                    }//if
#else
                    circuit_pc->wirep[n].in_ptr[i] = random_element(nodes_per_piece);
                    if ((100 * RAND_NUM) < pct_wire_in_piece) {
                        circuit_pc->wirep[n].out_ptr[i] = random_element(nodes_per_piece);
                    }//if
#endif
                    else {
#ifdef _DEBUG
                        cout << "\t\tShared appear\n";
#endif
                        // make wire as shared
                        circuit_pc->wirep[n].wire_attr[i] = 1;
                        int nn = int(drand48() * (num_pieces - 1));
                        if (nn >= n) nn++;
                        // pick an arbitrary node, except that if it's one that didn't used to be shared, make the 
                        //  sequentially next pointer shared instead so that each node's shared pointers stay compact
                        int idx = int(drand48() * nodes_per_piece);
                        if (idx > shared_nodes_piece[nn])
                            idx = shared_nodes_piece[nn]++;
                        // mark idx node of this piece the shr piece info 
                        circuit_pc->wirep[n].shr_pc[i] = nn;
                        // make output node as shared and record shared peieces
                        circuit_pc->nodep[nn].shr_pc[idx]    = n;
                        circuit_pc->nodep[nn].node_attr[idx] = 1;
                        circuit_pc->wirep[n].out_ptr[i] = idx;
                    }//else
#if 0
                   CkPrintf("\t**node info **\n");
                   CkPrintf("\tin_charge: %f, out_charge: %f\n", circuit_pc->nodep[n].charge[circuit_pc->wirep[n].in_ptr[i]], circuit_pc->nodep[n].charge[circuit_pc->wirep[n].out_ptr[i]]);
#endif
#ifdef _DEBUG
                   // circuit info
//                   CkPrintf( "Wire %d resistance: %f, inductance: %f, capacitance: %f\n", i, circuit_pc->wirep[n].resistance[i], circuit_pc->wirep[n].inductance[i], circuit_pc->wirep[n].capacitance[i]);
//                   CkPrintf("** node info **\n");
//                   CkPrintf("in_ptr/node_type:%d, capacitance: %f\n", circuit_pc->nodep[n].node_attr[(circuit_pc->wirep[n].in_ptr[i])], circuit_pc->nodep[n].capacitance[(circuit_pc->wirep[n].in_ptr[i])]);
//                   CkPrintf("out_ptr/node_type:%d, capacitance: %f\n", circuit_pc->nodep[n].node_attr[(circuit_pc->wirep[n].out_ptr[i])], circuit_pc->nodep[n].capacitance[(circuit_pc->wirep[n].out_ptr[i])]);
#endif
                }//for: wire_per_piece
            }//for : pieces
            //init transfer buffer size
            int pieces_per_pe = num_pieces / max_pe;
            transfer_size = sizeof(int) + pieces_per_pe * sizeof(PRECISION) * (nodes_per_piece + wires_per_piece);
            transfer_buf  = (unsigned char *) malloc(transfer_size);
            result_size   = sizeof(int) + 2*pieces_per_pe * sizeof(PRECISION) * nodes_per_piece;
            result_buf    = (unsigned char *) malloc(result_size);
            post_num = num_pieces * nodes_per_piece;
            post_charge = (PRECISION *)malloc(post_num*sizeof(PRECISION));  
            post_shr_num = num_pieces * wires_per_piece;
            post_shr_voltage = (PRECISION *)malloc(post_shr_num*sizeof(PRECISION));
        }// endif
        else {
#ifdef DLB
            usesAtSync = true;    
#endif
            //init transfer buffer size
            transfer_size = sizeof(int) + num_pieces * sizeof(PRECISION) * (nodes_per_piece + wires_per_piece);
            transfer_buf  = (unsigned char *) malloc(transfer_size);
            result_size   = sizeof(int) + 2*num_pieces * sizeof(PRECISION) * nodes_per_piece;
            result_buf    = (unsigned char *) malloc(result_size);
            post_num = max_pe * num_pieces * nodes_per_piece;
            post_charge = (PRECISION *)malloc(post_num*sizeof(PRECISION));  
            post_shr_num = max_pe * num_pieces * wires_per_piece;
            post_shr_voltage = (PRECISION *)malloc(post_shr_num*sizeof(PRECISION));
        }
#endif
    }
#if 1
    void update_post(CkReductionMsg *msg) {
        CkPrintf("My rank is: %d, update_post\n", thisIndex);
        int pieces_per_pe = num_pieces / max_pe;
        CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
        while(current != NULL) {
            unsigned char *result = (unsigned char *) &current->data;
            int * rev_id = reinterpret_cast<int *>(result);
            result += sizeof(int);
            PRECISION * rev_data = reinterpret_cast<float *>(result);
            //post work init
            for (int n=0; n<pieces_per_pe; n++) { 
                int noffset = (*rev_id) * pieces_per_pe + n;
                memcpy(circuit_pc->nodep[noffset].charge, rev_data+n*(nodes_per_piece+wires_per_piece), nodes_per_piece*sizeof(PRECISION));
                memcpy(circuit_pc->wirep[noffset].shr_charge, rev_data+n*(nodes_per_piece+wires_per_piece)+nodes_per_piece, wires_per_piece*sizeof(PRECISION));
            }
            current = current->next();
        }
        // post work to update node charge 
        for (int n = 0; n < num_pieces; n++) {
            for (int i=0; i<wires_per_piece; i++) {
                if (circuit_pc->wirep[n].wire_attr[i] == 1) {
                    circuit_pc->nodep[circuit_pc->wirep[n].shr_pc[i]].charge[circuit_pc->wirep[n].out_ptr[i]] += circuit_pc->wirep[n].shr_charge[i];
                    circuit_pc->wirep[n].shr_charge[i] = 0.f;
                }
            }
        } //end for num_pieces
        //// post_work to working PEs
        for (int p=0; p<max_pe; p++) {
            for (int n=0; n<pieces_per_pe; n++) {
                int disp = p * pieces_per_pe * nodes_per_piece + n * nodes_per_piece;
                int n_disp = p * pieces_per_pe + n;
                memcpy(post_charge+disp, circuit_pc->nodep[n_disp].charge, nodes_per_piece*sizeof(PRECISION));
            }
        }
#ifdef _DEBUG 
        for (int n=0; n<num_pieces; n++) {
            for (int it=0; it<nodes_per_piece; it++) {
                CkPrintf("\t**node info **\n");
                CkPrintf("\tvoltage: %f, charge: %f\n", circuit_pc->nodep[n].voltage[it], circuit_pc->nodep[n].charge[it]);
            }
        }
#endif
        CkPrintf("update_post complete!--->\n");  
        mainProxy.post_run();
        // send post_work to working PEs
        //SendPost(workPE, post_charge, post_num);
//DEBUG_W
        // working PEs receive post data 
        //printmain(10);
    }
#endif
    void update_res(CkReductionMsg *msg) {
        CkPrintf("My rank is: %d, update_res\n", thisIndex);
        CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
        int pieces_per_pe = num_pieces / max_pe;
        while(current != NULL) {
            int *result = (int *) &current->data;
            int rev_id = result[0];
            PRECISION * rev_data = (PRECISION *)((unsigned char*)result + sizeof(int));
            //post work init
            for (int n=0; n<pieces_per_pe; n++) { 
                int noffset = rev_id * pieces_per_pe + n;
                memcpy(circuit_pc->nodep[noffset].voltage, &rev_data[n*nodes_per_piece*2], nodes_per_piece*sizeof(PRECISION));
                memcpy(circuit_pc->nodep[noffset].charge , &rev_data[n*nodes_per_piece*2+nodes_per_piece], nodes_per_piece*sizeof(PRECISION));
            }
            current = current->next();
        }
        // post work to update shared voltage
        for (int n=0; n<num_pieces; n++) {
            for (int i=0; i<wires_per_piece; i++) {
                if (circuit_pc->wirep[n].wire_attr[i] == 1) {
                   circuit_pc->wirep[n].shr_voltage[i] =  circuit_pc->nodep[circuit_pc->wirep[n].shr_pc[i]].voltage[circuit_pc->wirep[n].out_ptr[i]];
                }
            }
        }
        // post_work to working PEs
        for (int p=0; p<max_pe; p++) {
            for (int n=0; n<pieces_per_pe; n++) {
                int disp = p * pieces_per_pe * wires_per_piece + n * wires_per_piece;
                int n_disp = p * pieces_per_pe + n;
                memcpy(post_shr_voltage+disp, circuit_pc->wirep[n_disp].shr_voltage, wires_per_piece*sizeof(PRECISION));
            }
        }
#if 1
        CkPrintf("\t**node info **\n");
        for (int n=0; n<num_pieces; n++) {
            for (int it=0; it<nodes_per_piece; it++) {
                CkPrintf("\tvoltage: %f, charge: %f\n", circuit_pc->nodep[n].voltage[it], circuit_pc->nodep[n].charge[it]);
            }
        }
#endif
        CkPrintf("update_res complete!--->\n");  
        mainProxy.done();
    }
    void pup(PUP::er &p) {
        p|mem_pc_size; p|mem_size; p|transfer_size; p|ctype;
        p|result_size; p|num_pieces; p|post_num; p|post_shr_num;
        if (p.isUnpacking()) {
            CkPrintf("\tElement %d unpacking on PE %d\n", thisIndex, CkMyPe());
            transfer_buf  = (unsigned char *) malloc(transfer_size);
            result_buf    = (unsigned char *) malloc(result_size);
            post_charge   = (PRECISION *)malloc(post_num*sizeof(PRECISION));
            post_shr_voltage = (PRECISION *)malloc(post_shr_num*sizeof(PRECISION));
            circuit_pc = new cct(num_pieces, wires_per_piece, nodes_per_piece, mem_pc_size, mem_size);
        }
        circuit_pc->pup(p);
        PUParray(p, transfer_buf , transfer_size);
        PUParray(p, result_buf, result_size);
        PUParray(p, post_charge, post_num);
        PUParray(p, post_shr_voltage, post_shr_num);
        CkPrintf("\tElement %d Complete transfer on PE %d\n", thisIndex, CkMyPe());
    }
    Grid(CkMigrateMessage*) {

    }
#ifdef DLB
    void pauseForLB() {
        CkPrintf("Element %d pause for LB on PE %d\n", thisIndex, CkMyPe());
        AtSync();
    } 
    void ResumeFromSync() {
        CkPrintf("Element %d Resume from Sync on PE %d, transfer_size:%d, result_size:%d, num_pieces:%d, post_num:%d, post_shr_num:%d\n", thisIndex, CkMyPe(), transfer_size, result_size, num_pieces, post_num, post_shr_num);
        //workPE.pgmrun(CkCallback(CkReductionTarget(Grid, update_post), mainPE));
        //CkCallback cbld(CkIndex_Main::resumeIter(), mainProxy);
        CkCallback cbld(CkReductionTarget(Main, resumeIter), mainProxy);
        contribute(sizeof(bool), &ctype, CkReduction::set, cbld);
    }

#endif
};

#include "circuit.def.h"
