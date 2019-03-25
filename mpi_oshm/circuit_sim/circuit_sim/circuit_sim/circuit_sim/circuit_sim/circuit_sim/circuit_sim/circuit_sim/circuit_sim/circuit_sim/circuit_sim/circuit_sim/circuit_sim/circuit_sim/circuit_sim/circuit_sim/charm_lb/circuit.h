#ifndef CIRCUIT_H_
#define CIRCUIT_H_

#include "circuitcuda.h"
#include "pup.h"
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


/**
 * @ structure of node and wire array
 */

//CProxy_Main mainProxy;

class cct {
public:
    typedef struct point {
       PRECISION * capacitance;
       PRECISION * leakage;
       PRECISION * charge;
       PRECISION * voltage;
       int       * shr_pc;
       int       * node_attr;
       void pup(PUP::er &p) {
       }
    } node;
    
    typedef struct edge {
        PRECISION ** currents;
        PRECISION ** voltages;
        PRECISION *  resistance;
        PRECISION *  inductance;
        PRECISION *  capacitance;
        PRECISION *  shr_voltage;
        PRECISION *  shr_charge;
        int       *  shr_pc;
        int       *  in_ptr;
        int       *  out_ptr;
        int       *  wire_attr;
        void pup(PUP::er &p) {
        }
    } wire;

    int    num_pc;
    int    nodes_per_pc;
    int    wires_per_pc;
    int    mem_size;
    int    mem_pc_size;
    unsigned char * mem_pool, * mem_begin;
    node * nodep;
    wire * wirep;
    
    cct();
    cct(int nup, int wp, int np, int mpc_size, int m_size) {
        num_pc = nup;
        nodes_per_pc = np;
        wires_per_pc = wp;
        mem_size     = m_size;
        mem_pc_size  = mpc_size;
        nodep = new node[num_pc];
        wirep = new wire[num_pc];
        mem_pool = (unsigned char *)malloc(mem_size);
        mem_begin = mem_pool;
        cct_init();
    }
    void cct_init() {
        for (int n=0; n<num_pc; n++) {
            // allocate space for array in soa
            nodep[n].capacitance = reinterpret_cast<PRECISION *>(mem_begin);
            mem_begin += sizeof(PRECISION) * nodes_per_pc;
            nodep[n].leakage     = reinterpret_cast<PRECISION *>(mem_begin);
            mem_begin += sizeof(PRECISION) * nodes_per_pc;
            nodep[n].charge      = reinterpret_cast<PRECISION *>(mem_begin);
            mem_begin += sizeof(PRECISION) * nodes_per_pc;
            nodep[n].voltage     = reinterpret_cast<PRECISION *>(mem_begin);
            mem_begin += sizeof(PRECISION) * nodes_per_pc;
            nodep[n].shr_pc      = reinterpret_cast<int *>(mem_begin);
            mem_begin += sizeof(int) * nodes_per_pc;
            nodep[n].node_attr   = reinterpret_cast<int *>(mem_begin);
            mem_begin += sizeof(int) * nodes_per_pc;
            // allocate space for array in soa of wire
            wirep[n].currents    = (PRECISION **)new PRECISION*[wires_per_pc];
            for (int j=0; j<wires_per_pc; j++)
                wirep[n].currents[j]  = reinterpret_cast<PRECISION *>(&mem_begin[j*WIRE_SEGMENTS*sizeof(PRECISION)]);
            mem_begin += sizeof(PRECISION) * wires_per_pc * WIRE_SEGMENTS;
            wirep[n].voltages    = (PRECISION **)new PRECISION*[wires_per_pc];
            for (int j=0; j<wires_per_pc; j++)
                wirep[n].voltages[j]  = reinterpret_cast<PRECISION *>(&mem_begin[j*(WIRE_SEGMENTS-1)*sizeof(PRECISION)]);
            mem_begin += sizeof(PRECISION) * wires_per_pc * (WIRE_SEGMENTS-1);
            wirep[n].resistance  = reinterpret_cast<PRECISION *>(mem_begin);
            mem_begin += sizeof(PRECISION) * wires_per_pc;
            wirep[n].inductance  = reinterpret_cast<PRECISION *>(mem_begin);
            mem_begin += sizeof(PRECISION) * wires_per_pc;
            wirep[n].capacitance = reinterpret_cast<PRECISION *>(mem_begin);
            mem_begin += sizeof(PRECISION) * wires_per_pc;
            wirep[n].in_ptr      = reinterpret_cast<int *>(mem_begin);
            mem_begin += sizeof(int) * wires_per_pc;
            wirep[n].out_ptr     = reinterpret_cast<int *>(mem_begin);
            mem_begin += sizeof(int) * wires_per_pc;
            wirep[n].wire_attr   = reinterpret_cast<int *>(mem_begin);
            mem_begin += sizeof(int) * wires_per_pc;
            // init wire shared part
            wirep[n].shr_voltage = reinterpret_cast<PRECISION *>(mem_begin);
            mem_begin += sizeof(PRECISION) * wires_per_pc;
            wirep[n].shr_charge  = reinterpret_cast<PRECISION *>(mem_begin);
            mem_begin += sizeof(PRECISION) * wires_per_pc;
            wirep[n].shr_pc      = reinterpret_cast<int *>(mem_begin);
            mem_begin += sizeof(int) * wires_per_pc;
        }
    }
    void pup(PUP::er &p) {
        p|mem_pc_size; p|mem_size; 
        p|num_pc; p|nodes_per_pc; p|wires_per_pc;
        if (p.isUnpacking()) {
         //   circuit_pc->nodep  = new node[num_pieces];
         //   circuit_pc->wirep = new wire[num_pieces];
            nodep = new node[num_pc];
            wirep = new wire[num_pc];
            mem_pool = (unsigned char *)malloc(mem_size);
        }
        PUParray(p, mem_pool  , mem_size  );
        mem_begin = mem_pool;
        cct_init();
    }

};
#endif
