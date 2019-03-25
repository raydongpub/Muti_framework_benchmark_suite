#include "circuit.decl.h"
#include "circuit.h"
#include "stdlib.h"
#include <vector>
#include <unistd.h>
#include <sys/time.h>

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
 * @ structure of node and wire array
 */
//struct point;
//typedef struct point node;
//struct point {
//   PRECISION * capacitance;
//   PRECISION * leakage;
//   PRECISION * charge;
//   PRECISION * voltage;
//   int       * shr_pc;
//   int       * node_attr;
//};
//
//struct edge;
//typedef struct edge wire;
//struct edge {
//    PRECISION ** currents;
//    PRECISION ** voltages;
//    PRECISION *  resistance;
//    PRECISION *  inductance;
//    PRECISION *  capacitance;
//    PRECISION *  shr_voltage;
//    PRECISION *  shr_charge;
//    int       *  shr_pc;
//    int       *  in_ptr;
//    int       *  out_ptr;
//    int       *  wire_attr;
//};
/**
 * @ Functions
 */

void init_val(int num, unsigned char *dst, unsigned char *src) {
    memcpy(dst, src, num);
} 
void init_post(node * node_pc, PRECISION * src, int peid, int nodes_per_pc, int num_pc) {
    for (int i=0; i<num_pc; i++) {
        int disp = peid * num_pc * nodes_per_pc + i * nodes_per_pc;
        memcpy(node_pc[i].charge, &src[disp], nodes_per_pc * sizeof(PRECISION));
    }
}
void init_post_shr(wire * wire_pc, PRECISION * src, int peid, int wires_per_pc, int num_pc) {
    for (int i=0; i<num_pc; i++) {
        int disp = peid * num_pc * wires_per_pc + i * wires_per_pc;
        memcpy(wire_pc[i].shr_voltage, &src[disp], wires_per_pc * sizeof(PRECISION));
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
// cuda functions
extern void cudaRun(node * node_pc, wire * wire_pc, unsigned char * buf, int nodes_per_pc, int wires_per_pc, int pieces_per_pe, int peid, int gpu_block, int gpu_thread);
extern void cudaPost(node * node_pc, wire * wire_pc, unsigned char * buf, int nodes_per_pc, int wires_per_pc, int pieces_per_pe, int peid, int gpu_block, int gpu_thread);
void cudaInit(bool first, node * node_pc, wire * wire_pc, unsigned char * buf, int nodes_per_pc, int wires_per_pc, int pieces_per_pe, int peid, int gpu_block, int gpu_thread) {
    if (first)
        cudaRun(node_pc, wire_pc, buf, nodes_per_pc, wires_per_pc, pieces_per_pe, peid, gpu_block, gpu_thread);
    else
        cudaPost(node_pc, wire_pc, buf, nodes_per_pc, wires_per_pc, pieces_per_pe, peid, gpu_block, gpu_thread);
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
#if 0
        workPE.printout(2, CkCallback(CkReductionTarget(Grid, update_res), mainPE));
#endif
        workPE.pgmrun(CkCallback(CkReductionTarget(Grid, update_post), mainPE));
        //workPE.printout(2, CkCallback(CkReductionTarget(Main, done), mainPE));
        //mainPE.printmain(10);
    }
    void post_run() {
        CkPrintf("SendPost--->\n");
        mainPE.SendPost(workPE);
        CkPrintf("SendPost complete--->\n");
        workPE.postrun(CkCallback(CkReductionTarget(Grid, update_res), mainPE));
    }
    void done() {
        loop_count++;
        if (loop_count < number_loops) {
            mainPE.SendLoop(workPE);
            workPE.pgmrunloop(CkCallback(CkReductionTarget(Grid, update_post), mainPE));
        }
        else {
            mainPE.cleanup();
            workPE.cleanup();
            CkExit();
        }
    }
};

class Grid : public CBase_Grid {
    node * node_piece;
    wire * wire_piece;
    PRECISION * post_charge, * post_shr_voltage;
    unsigned char * mem_pool, * mem_begin, * transfer_buf, * result_buf;
    int mem_pc_size, mem_size, transfer_size, result_size; 
    int num_pieces;
    int post_num, post_shr_num;
    Grid_SDAG_CODE
    public:
    Grid(bool accept, int num_pieces_) : num_pieces(num_pieces_){
        CkPrintf("Rank: %d, [num_pieces:%d], [num_nodes:%d], [num_wires:%d], [num_loops:%d], [max_pe:%d]\n", thisIndex, num_pieces, nodes_per_piece, wires_per_piece, number_loops, max_pe);
        node_piece  = new node[num_pieces];
        // wire definition
        wire_piece = new wire[num_pieces];
        mem_pc_size = nodes_per_piece * (sizeof(PRECISION) * 4 + sizeof(int) * 2)
                 + wires_per_piece * (sizeof(PRECISION) * (2 * WIRE_SEGMENTS + 4) + sizeof(int) * 4);
        mem_size = num_pieces * mem_pc_size;
        mem_pool = (unsigned char *)malloc(mem_size);
        mem_begin = mem_pool;
        for (int n=0; n<num_pieces; n++) {
            // allocate space for array in soa
            node_piece[n].capacitance = reinterpret_cast<PRECISION *>(mem_pool);
            mem_pool += sizeof(PRECISION) * nodes_per_piece;
            node_piece[n].leakage     = reinterpret_cast<PRECISION *>(mem_pool);
            mem_pool += sizeof(PRECISION) * nodes_per_piece;
            node_piece[n].charge      = reinterpret_cast<PRECISION *>(mem_pool);
            mem_pool += sizeof(PRECISION) * nodes_per_piece;
            node_piece[n].voltage     = reinterpret_cast<PRECISION *>(mem_pool);
            mem_pool += sizeof(PRECISION) * nodes_per_piece;
            node_piece[n].shr_pc      = reinterpret_cast<int *>(mem_pool);
            mem_pool += sizeof(int) * nodes_per_piece;
            node_piece[n].node_attr   = reinterpret_cast<int *>(mem_pool);
            mem_pool += sizeof(int) * nodes_per_piece;
            // allocate space for array in soa of wire
            wire_piece[n].currents    = (PRECISION **)new PRECISION*[wires_per_piece];
            for (int j=0; j<wires_per_piece; j++)
                wire_piece[n].currents[j]  = reinterpret_cast<PRECISION *>(&mem_pool[j*WIRE_SEGMENTS*sizeof(PRECISION)]);
            mem_pool += sizeof(PRECISION) * wires_per_piece * WIRE_SEGMENTS;
            wire_piece[n].voltages    = (PRECISION **)new PRECISION*[wires_per_piece];
            for (int j=0; j<wires_per_piece; j++)
                wire_piece[n].voltages[j]  = reinterpret_cast<PRECISION *>(&mem_pool[j*(WIRE_SEGMENTS-1)*sizeof(PRECISION)]);
            mem_pool += sizeof(PRECISION) * wires_per_piece * (WIRE_SEGMENTS-1);
            wire_piece[n].resistance  = reinterpret_cast<PRECISION *>(mem_pool);
            mem_pool += sizeof(PRECISION) * wires_per_piece;
            wire_piece[n].inductance  = reinterpret_cast<PRECISION *>(mem_pool);
            mem_pool += sizeof(PRECISION) * wires_per_piece;
            wire_piece[n].capacitance = reinterpret_cast<PRECISION *>(mem_pool);
            mem_pool += sizeof(PRECISION) * wires_per_piece;
            wire_piece[n].in_ptr      = reinterpret_cast<int *>(mem_pool);
            mem_pool += sizeof(int) * wires_per_piece;
            wire_piece[n].out_ptr     = reinterpret_cast<int *>(mem_pool);
            mem_pool += sizeof(int) * wires_per_piece;
            wire_piece[n].wire_attr   = reinterpret_cast<int *>(mem_pool);
            mem_pool += sizeof(int) * wires_per_piece;
            // init wire shared part
            wire_piece[n].shr_voltage = reinterpret_cast<PRECISION *>(mem_pool);
            mem_pool += sizeof(PRECISION) * wires_per_piece;
            wire_piece[n].shr_charge  = reinterpret_cast<PRECISION *>(mem_pool);
            mem_pool += sizeof(PRECISION) * wires_per_piece;
            wire_piece[n].shr_pc      = reinterpret_cast<int *>(mem_pool);
            mem_pool += sizeof(int) * wires_per_piece;
        }
#if 1
        if (!accept) {
            // set random seed
            srand48(random_seed);
            std::vector<int> shared_nodes_piece(num_pieces, 0);
            // node initialization
            for (int n = 0; n < num_pieces; n++) {
                // initialize node_per_piece
                for (int i = 0; i < nodes_per_piece; i++) {
                    // initialize node parameter
#ifndef D_RAND
                    node_piece[n].capacitance[i] = drand48() + 1.f;
                    node_piece[n].leakage[i]     = 0.1f * drand48();
                    node_piece[n].charge[i]      = 0.f;
                    node_piece[n].voltage[i]     = 2*drand48() - 1.f;
#else
                    node_piece[n].capacitance[i] = RAND_NUM + 1.f;
                    node_piece[n].leakage[i]     = 0.1f * RAND_NUM;
                    node_piece[n].charge[i]      = 0.f;
                    node_piece[n].voltage[i]     = 2*RAND_NUM - 1.f;
#endif
                    // node_attr (0:private, 1:shared)
                    node_piece[n].shr_pc[i]      = 0;
                    node_piece[n].node_attr[i]   = 0;
                    CkPrintf("\tvoltage: %f, charge: %f\n", node_piece[n].voltage[i], node_piece[n].charge[i]);
                }//for
            }//for

            // wire initialization
            for (int n = 0; n < num_pieces; n++) {
#ifdef _DEBUG
                CkPrintf("=== List nodes in piece %d ===\n", n);
#endif
                for (int j=0; j<wires_per_piece; j++) {
                    wire_piece[n].shr_voltage[j] = 0.f;
                    wire_piece[n].shr_charge[j]  = 0.f;
                    wire_piece[n].shr_pc[j]      = 0;
                    wire_piece[n].wire_attr[j]   = 0;
                }
                // initialize wire parameter
                for (int i = 0; i < wires_per_piece; i++) {
                    // init currents
                    for (int j = 0; j < WIRE_SEGMENTS; j++)
                        wire_piece[n].currents[i][j] = 0.f;
                    // init voltage
                    for (int j = 0; j < WIRE_SEGMENTS-1; j++)
                        wire_piece[n].voltages[i][j] = 0.f;
#ifndef D_RAND
                    // init resistance
                    wire_piece[n].resistance[i]  = drand48() * 10.0 + 1.0;
                    // Keep inductance on the order of 1e-3 * dt to avoid resonance problems
                    wire_piece[n].inductance[i]  = (drand48() + 0.1) * DELTAT * 1e-3;
                    wire_piece[n].capacitance[i] = drand48() * 0.1;
#else
                    // init resistance
                    wire_piece[n].resistance[i]  = RAND_NUM * 10.0 + 1.0;
                    // Keep inductance on the order of 1e-3 * dt to avoid resonance problems
                    wire_piece[n].inductance[i]  = (RAND_NUM + 0.1) * DELTAT * 1e-3;
                    wire_piece[n].capacitance[i] = RAND_NUM * 0.1;

#endif
                    // UNC init connection
#ifndef D_RAND
                    wire_piece[n].in_ptr[i] = random_element(nodes_per_piece);
                    if ((100 * drand48()) < pct_wire_in_piece) {
                        wire_piece[n].out_ptr[i] = random_element(nodes_per_piece);
                    }//if
#else
                    wire_piece[n].in_ptr[i] = random_element(nodes_per_piece);
                    if ((100 * RAND_NUM) < pct_wire_in_piece) {
                        wire_piece[n].out_ptr[i] = random_element(nodes_per_piece);
                    }//if
#endif
                    else {
#ifdef _DEBUG
                        cout << "\t\tShared appear\n";
#endif
                        // make wire as shared
                        wire_piece[n].wire_attr[i] = 1;
                        int nn = int(drand48() * (num_pieces - 1));
                        if (nn >= n) nn++;
                        // pick an arbitrary node, except that if it's one that didn't used to be shared, make the 
                        //  sequentially next pointer shared instead so that each node's shared pointers stay compact
                        int idx = int(drand48() * nodes_per_piece);
                        if (idx > shared_nodes_piece[nn])
                            idx = shared_nodes_piece[nn]++;
                        // mark idx node of this piece the shr piece info 
                        wire_piece[n].shr_pc[i] = nn;
                        // make output node as shared and record shared peieces
                        node_piece[nn].shr_pc[idx]    = n;
                        node_piece[nn].node_attr[idx] = 1;
                        wire_piece[n].out_ptr[i] = idx;
                    }//else
#if 0
                   CkPrintf("\t**node info **\n");
                   CkPrintf("\tin_charge: %f, out_charge: %f\n", node_piece[n].charge[wire_piece[n].in_ptr[i]], node_piece[n].charge[wire_piece[n].out_ptr[i]]);
#endif
#ifdef _DEBUG
                   // circuit info
                   CkPrintf( "Wire %d resistance: %f, inductance: %f, capacitance: %f\n", i, wire_piece[n].resistance[i], wire_piece[n].inductance[i], wire_piece[n].capacitance[i]);
//                   CkPrintf("** node info **\n");
//                   CkPrintf("in_ptr/node_type:%d, capacitance: %f\n", node_piece[n].node_attr[(wire_piece[n].in_ptr[i])], node_piece[n].capacitance[(wire_piece[n].in_ptr[i])]);
//                   CkPrintf("out_ptr/node_type:%d, capacitance: %f\n", node_piece[n].node_attr[(wire_piece[n].out_ptr[i])], node_piece[n].capacitance[(wire_piece[n].out_ptr[i])]);
#endif
                }//for: wire_per_piece
            }//for : pieces
            //init transfer buffer size
            int pieces_per_pe = num_pieces / max_pe;
            transfer_size = sizeof(int) + pieces_per_pe * sizeof(PRECISION) * (nodes_per_piece + wires_per_piece);
            transfer_buf  = (unsigned char *) malloc(transfer_size);
            result_size   = sizeof(int) + 2*pieces_per_pe * sizeof(PRECISION) * nodes_per_piece;
            result_buf    = (unsigned char *) malloc(result_size);
        }// endif
        else {
            //init transfer buffer size
            transfer_size = sizeof(int) + num_pieces * sizeof(PRECISION) * (nodes_per_piece + wires_per_piece);
            transfer_buf  = (unsigned char *) malloc(transfer_size);
            result_size   = sizeof(int) + 2*num_pieces * sizeof(PRECISION) * nodes_per_piece;
            result_buf    = (unsigned char *) malloc(result_size);
        }
#endif
    }
#if 1
    void update_post(CkReductionMsg *msg) {
        CkPrintf("My rank is: %d, update_post\n", thisIndex);
        CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
        int pieces_per_pe = num_pieces / max_pe;
        while(current != NULL) {
            unsigned char *result = (unsigned char *) &current->data;
            int * rev_id = reinterpret_cast<int *>(result);
            result += sizeof(int);
            PRECISION * rev_data = reinterpret_cast<float *>(result);
            //post work init
            for (int n=0; n<pieces_per_pe; n++) { 
                int noffset = (*rev_id) * pieces_per_pe + n;
                memcpy(node_piece[noffset].charge, rev_data+n*(nodes_per_piece+wires_per_piece), nodes_per_piece*sizeof(PRECISION));
                memcpy(wire_piece[noffset].shr_charge, rev_data+n*(nodes_per_piece+wires_per_piece)+nodes_per_piece, wires_per_piece*sizeof(PRECISION));
            }
            current = current->next();
        }
        // post work to update node charge 
        for (int n = 0; n < num_pieces; n++) {
            for (int i=0; i<wires_per_piece; i++) {
                if (wire_piece[n].wire_attr[i] == 1) {
                    node_piece[wire_piece[n].shr_pc[i]].charge[wire_piece[n].out_ptr[i]] += wire_piece[n].shr_charge[i];
                    wire_piece[n].shr_charge[i] = 0.f;
                }
            }
        } //end for num_pieces
        // post_work to working PEs
        post_num = max_pe * pieces_per_pe * nodes_per_piece;
        post_charge = (PRECISION *)malloc(post_num*sizeof(PRECISION));  
        for (int p=0; p<max_pe; p++) {
            for (int n=0; n<pieces_per_pe; n++) {
                int disp = p * pieces_per_pe * nodes_per_piece + n * nodes_per_piece;
                int n_disp = p * pieces_per_pe + n;
                memcpy(post_charge+disp, node_piece[n_disp].charge, nodes_per_piece*sizeof(PRECISION));
            }
        }
#define _DEBUG
#ifdef _DEBUG 
        for (int n=0; n<num_pieces; n++) {
            for (int it=0; it<nodes_per_piece; it++) {
                CkPrintf("\t**node info **\n");
                CkPrintf("\tvoltage: %f, charge: %f\n", node_piece[n].voltage[it], node_piece[n].charge[it]);
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
            unsigned char *result = (unsigned char *) &current->data;
            int * rev_id = reinterpret_cast<int *>(result);
            result += sizeof(int);
            PRECISION * rev_data = reinterpret_cast<float *>(result);
            //post work init
            for (int n=0; n<pieces_per_pe; n++) { 
                int noffset = (*rev_id) * pieces_per_pe + n;
                memcpy(node_piece[noffset].voltage, rev_data+n*nodes_per_piece*2, nodes_per_piece*sizeof(PRECISION));
                memcpy(node_piece[noffset].charge , rev_data+n*nodes_per_piece*2 + nodes_per_piece, nodes_per_piece*sizeof(PRECISION));
            }
            current = current->next();
        }
        // post work to update shared voltage
        for (int n=0; n<num_pieces; n++) {
            for (int i=0; i<wires_per_piece; i++) {
                if (wire_piece[n].wire_attr[i] == 1) {
                   wire_piece[n].shr_voltage[i] =  node_piece[wire_piece[n].shr_pc[i]].voltage[wire_piece[n].out_ptr[i]];
                }
            }
        }
        // post_work to working PEs
        post_shr_num = max_pe * pieces_per_pe * wires_per_piece;
        post_shr_voltage = (PRECISION *)malloc(post_shr_num*sizeof(PRECISION));
        for (int p=0; p<max_pe; p++) {
            for (int n=0; n<pieces_per_pe; n++) {
                int disp = p * pieces_per_pe * wires_per_piece + n * wires_per_piece;
                int n_disp = p * pieces_per_pe + n;
                memcpy(post_shr_voltage+disp, wire_piece[n_disp].shr_voltage, wires_per_piece*sizeof(PRECISION));
            }
        }
#if 1
        for (int n=0; n<num_pieces; n++) {
            for (int it=0; it<nodes_per_piece; it++) {
                CkPrintf("\t**node info **\n");
                CkPrintf("\tvoltage: %f, charge: %f\n", node_piece[n].voltage[it], node_piece[n].charge[it]);
            }
        }
#endif
        CkPrintf("update_res complete!--->\n");  
        mainProxy.done();
    }
    void cleanup() {
        free(mem_begin);
        free(transfer_buf);
        free(result_buf);
        free(post_charge);
    }
    Grid(CkMigrateMessage*) {

    }

};

#include "circuit.def.h"
