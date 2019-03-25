#ifndef CIRCUIT_H_
#define CIRCUIT_H_
/**
 * @ data type definition
 */
#define PRECISION float
#define WIRE_SEGMENTS 10
#define STEPS         10000
#define DELTAT        1e-6

#define INDEX_TYPE    unsigned
#define INDEX_DIM     1

#define D_NODE        0x0000
#define D_WIRE        0x0001

/**
 * @ structure of node and wire array
 */
struct point;
typedef struct point node;
struct point {
   PRECISION * capacitance;
   PRECISION * leakage;
   PRECISION * charge;
   PRECISION * voltage;
   int       * shr_pc;
   int       * node_attr;
};

struct edge;
typedef struct edge wire;
struct edge {
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
};

#endif
