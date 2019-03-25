#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <iostream>
#include <vector>

using namespace std;

#define _RNODE

/**
 * @ command line
 */
#define HELP_S         "-h"
#define HELP_L         "--help"
#define LOOP_S         "-l"
#define LOOP_L         "--loop"
#define PIECE_S        "-p"
#define PIECE_L        "--piece"
#define NODE_S         "-n"
#define NODE_L         "--node"
#define WIRE_S         "-w"
#define WIRE_L         "--wire"
#define PERCENT_S      "-c"
#define PERCENT_L      "-percent"
#define SEED_S         "-s"
#define SEED_L         "--seed"
#define BLOCK_S        "-b"
#define BLOCK_L        "--block"
#define THREAD_S       "-t"
#define THREAD_L       "--thread"

#define PE_S           "-pe"
#define PE_L           "--proc"
/**
 * @ data type definition
 */
#define PRECISION float
#define MPI_PRECISION MPI_FLOAT
#define WIRE_SEGMENTS 10
#define STEPS         10000
#define DELTAT        1e-6

#define INDEX_TYPE    unsigned
#define INDEX_DIM     1

#define D_NODE        0x0000
#define D_WIRE        0x0001

int num_loops, num_pieces, nodes_per_piece;
int wires_per_piece, pct_wire_in_piece;
int random_seed, num_blocks, num_threads, num_pe;
/**
 * @ check error function
 */
inline void checkError(int ret, const char * str) {
    if (ret != 0) {
        cerr << "Error: " << str << endl;
        exit(-1);
    }
}

inline void cudaCheckError(int line, cudaError_t ce)
{
    if (ce != cudaSuccess){
        printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
        exit(1);
    }
}

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

/**
 * @ Kernel Function
 */

// calculate currents gpu
__global__ void calculate_current_gpu(int num_wires, 
                PRECISION * wire_currents, PRECISION * wire_voltages, 
                int * in_ptr, int * out_ptr, 
                PRECISION * wire_inductance, PRECISION * wire_resistance, PRECISION * wire_capacitance, 
                PRECISION * node_voltage, int * wire_attr,
                PRECISION * shr_voltage) {
    int gridsize = gridDim.x * blockDim.x;
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_wires) {
        PRECISION temp_v[WIRE_SEGMENTS+1];
        PRECISION temp_i[WIRE_SEGMENTS];
        PRECISION old_i[WIRE_SEGMENTS];
        PRECISION old_v[WIRE_SEGMENTS-1];
        for (int it=idx; it<num_wires; it+=gridsize) {
            PRECISION dt = DELTAT;
            PRECISION recip_dt = 1.0f / dt;
            int steps = STEPS;
            int currents_offset = it * WIRE_SEGMENTS;
            int voltages_offset = it * (WIRE_SEGMENTS-1);
            // calc temporary variables
            for (int j = 0; j < WIRE_SEGMENTS; j++) {
                temp_i[j] = wire_currents[currents_offset+j];
                old_i[j]  = temp_i[j];
            }
            for (int j = 0; j < (WIRE_SEGMENTS-1); j++) {
                temp_v[j+1] = wire_voltages[voltages_offset+j];
                old_v[j]    = temp_v[j+1];
            }
            // calc outer voltages to the node voltages
            temp_v[0] = node_voltage[in_ptr[it]];
            // Note: out-ptr need communication when parallel
            if (wire_attr[it] == 0)
                temp_v[WIRE_SEGMENTS] = node_voltage[out_ptr[it]];
            else 
                temp_v[WIRE_SEGMENTS] = shr_voltage[it];
            // Solve the RLC model iteratively
            PRECISION inductance = wire_inductance[it];
            PRECISION recip_resistance = 1.0f / (wire_resistance[it]);
            PRECISION recip_capacitance = 1.0f / (wire_capacitance[it]);
            for (int j = 0; j < steps; j++) {
                // first, figure out the new current from the voltage differential
                // and our inductance:
                // dV = R*I + L*I' ==> I = (dV - L*I')/R
                for (int k = 0; k < WIRE_SEGMENTS; k++) {
                    temp_i[k] = ((temp_v[k+1] - temp_v[k]) - (inductance * (temp_i[k] - old_i[k]) * recip_dt)) * recip_resistance;
                }
                // Now update the inter-node voltages
                for (int k = 0; k < (WIRE_SEGMENTS-1); k++) {
                    temp_v[k+1] = old_v[k] + dt * (temp_i[k] - temp_i[k+1]) * recip_capacitance;
                }
            }
            // Write out the results
            for (int j = 0; j < WIRE_SEGMENTS; j++)
                wire_currents[currents_offset+j] = temp_i[j];
            for (int j = 0; j < (WIRE_SEGMENTS-1); j++)
                wire_voltages[voltages_offset+j] = temp_v[j+1];
        }// for: wires
    }// if
    __syncthreads();
}// calc_end

// distributed charge gpu
__global__ void distributed_charge_gpu(int num_wires, 
                PRECISION * wire_currents,  
                int * in_ptr, int * out_ptr, 
                PRECISION * node_charge, int * wire_attr,
                PRECISION * shr_charge) {
    int gridsize = gridDim.x * blockDim.x;
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_wires) {
        for (int it = idx; it < num_wires; it+=gridsize) {
            int currents_offset = it * WIRE_SEGMENTS;
            // calc temporary variables
            PRECISION dt = DELTAT;
            PRECISION in_current = -dt * (wire_currents[currents_offset]);
            PRECISION out_current = -dt * (wire_currents[currents_offset+WIRE_SEGMENTS-1]);
            //node_charge[in_ptr[it]]  += in_current;
            atomicAdd(&node_charge[in_ptr[it]], in_current);
            //node_charge[out_ptr[it]] += out_current;
            if (wire_attr[it] == 0)
                atomicAdd(&node_charge[out_ptr[it]], out_current);
            else
                atomicAdd(&shr_charge[it], out_current);
        }//for: iterate wires_per_piece
    }// if
    __syncthreads();
}// dc end
// update voltage gpu
__global__ void update_voltage_gpu( int num_nodes,
                PRECISION * node_voltage, PRECISION * node_charge, 
                PRECISION * node_capacitance, PRECISION * node_leakage) {
    int gridsize = gridDim.x * blockDim.x;
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        for (int it = idx; it < num_nodes; it+=gridsize) {
            PRECISION voltage = node_voltage[it];
            PRECISION charge = node_charge[it];
            PRECISION capacitance = node_capacitance[it];
            PRECISION leakage = node_leakage[it];
            voltage += charge / capacitance;
            voltage *= (1.f - leakage);
            //node_pc[n].voltage[it] = voltage;
            node_voltage[it] = voltage;
            node_charge[it]  = 0.f;
        }//for: iterate nodess_per_piece
    }//if
    __syncthreads();
}

/**
 * @ random function to get node
 */

int random_element(int vec_size)
{
  int index = int(drand48() * vec_size);
  return index;
}

void node_init(node input, int ioffset, node * output, int ooffset, int param_size) {
    for (int i=0; i<param_size; i++) {
        output[ooffset].capacitance = input.capacitance + i+ioffset;
        output[ooffset].leakage     = input.leakage     + i+ioffset;
        output[ooffset].charge      = input.charge      + i+ioffset;
        output[ooffset].voltage     = input.voltage     + i+ioffset;
        output[ooffset].node_attr   = input.node_attr   + i+ioffset;
    }
}
void wire_init(wire input, int ioffset, wire * output, int ooffset, int param_size) {
    for (int i=0; i<param_size; i++) {
        output[ooffset].currents    = input.currents + i+ioffset;
        output[ooffset].voltages    = input.voltages + i+ioffset;
        output[ooffset].resistance  = input.resistance + i+ioffset;
        output[ooffset].inductance  = input.inductance+ i+ioffset;
        output[ooffset].capacitance = input.capacitance + i+ioffset;
        output[ooffset].in_ptr      = input.in_ptr   + i+ioffset;
        output[ooffset].out_ptr     = input.out_ptr  + i+ioffset;
        output[ooffset].shr_pc      = input.shr_pc   + i+ioffset;
        output[ooffset].shr_voltage = input.shr_voltage + i+ioffset;
        output[ooffset].shr_charge  = input.shr_charge  + i+ioffset;
        output[ooffset].wire_attr   = input.wire_attr   + i+ioffset;

    }
}

/**
 * @ computation function
 */
void calculate_current_cpu(int num_pc, int num_wires, wire * wire_pc, node * node_pc) {

    // calculate currents
    for (int n=0; n < num_pc; n++) {
        // define temporaty variables
        PRECISION temp_v[WIRE_SEGMENTS+1];
        PRECISION temp_i[WIRE_SEGMENTS];
        PRECISION old_i[WIRE_SEGMENTS];
        PRECISION old_v[WIRE_SEGMENTS-1];
        // access first wire in each wire_piece
        //wire wire_head = first_wires[n];
        for (int it = 0 ; it < num_wires; ++it) {
#ifdef _DEBUG
            // circuit info
            printf("============Testify node: ============\n");
            printf( "Wire %d resistance: %f, inductance: %f, capacitance: %f\n", it, wire_pc[n].resistance[it], wire_pc[n].inductance[it], wire_pc[n].capacitance[it]);
            printf("** node info **\n");
            printf("in_ptr/node_type:%d, capacitance: %f\n", node_pc[n].node_attr[wire_pc[n].in_ptr[it]], node_pc[n].capacitance[wire_pc[n].in_ptr[it]]);
            printf("out_ptr/node_type:%d, capacitance: %f\n", node_pc[n].node_attr[wire_pc[n].out_ptr[it]],node_pc[n].capacitance[wire_pc[n].out_ptr[it]]);
#endif
            // define calc parameters
            PRECISION dt = DELTAT;
            PRECISION recip_dt = 1.0f / dt;
            int steps = STEPS;
            // calc temporary variables
            for (int i = 0; i < WIRE_SEGMENTS; i++) {
                temp_i[i] = wire_pc[n].currents[it][i];
                old_i[i]  = temp_i[i];
            }
            for (int i = 0; i < (WIRE_SEGMENTS-1); i++) {
                temp_v[i+1] = wire_pc[n].voltages[it][i];
                old_v[i]    = temp_v[i+1];
            }
            // calc outer voltages to the node voltages
            temp_v[0] = node_pc[n].voltage[wire_pc[n].in_ptr[it]];
            //temp_v[0] = *((wire_pc[n].in_ptr[it]).voltage);
            // Note: out-ptr need communication when parallel
            temp_v[WIRE_SEGMENTS] = node_pc[n].voltage[wire_pc[n].out_ptr[it]];
            //temp_v[WIRE_SEGMENTS] = *((wire_pc[n].out_ptr[it]).voltage);
            // Solve the RLC model iteratively
            PRECISION inductance = wire_pc[n].inductance[it];
            PRECISION recip_resistance = 1.0f / (wire_pc[n].resistance[it]);
            PRECISION recip_capacitance = 1.0f / (wire_pc[n].capacitance[it]);
            for (int j = 0; j < steps; j++) {
                // first, figure out the new current from the voltage differential
                // and our inductance:
                // dV = R*I + L*I' ==> I = (dV - L*I')/R
                for (int i = 0; i < WIRE_SEGMENTS; i++) {
                    temp_i[i] = ((temp_v[i+1] - temp_v[i]) - (inductance * (temp_i[i] - old_i[i]) * recip_dt)) * recip_resistance;
                }
                // Now update the inter-node voltages
                for (int i = 0; i < (WIRE_SEGMENTS-1); i++) {
                    temp_v[i+1] = old_v[i] + dt * (temp_i[i] - temp_i[i+1]) * recip_capacitance;
                }
            }
            // Write out the results
            for (int i = 0; i < WIRE_SEGMENTS; i++)
                wire_pc[n].currents[it][i] = temp_i[i];
            for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
                wire_pc[n].voltages[it][i] = temp_v[i+1];
       } //for: iterate wires_per_piece

    }// for: pieces, CN

}

void distributed_charge_cpu(int num_pc, int num_wires, wire * wire_pc, node * node_pc) {
    for (int n=0; n<num_pc; n++) {
        for (int it = 0; it < num_wires; ++it) {
            PRECISION dt = DELTAT;
            PRECISION in_current = -dt * (wire_pc[n].currents[it][0]);
            PRECISION out_current = -dt * (wire_pc[n].currents[it][WIRE_SEGMENTS-1]);
            node_pc[n].charge[wire_pc[n].in_ptr[it]]  += in_current;
            //*((wire_pc[n].in_ptr[it]).charge)  += in_current;
            node_pc[n].charge[wire_pc[n].out_ptr[it]] += out_current;
            //*((wire_pc[n].out_ptr[it]).charge) += out_current;

        }//for: iterate wires_per_piece
    }// for: pieces, DC
}

void update_voltage_cpu(int num_pc, int num_nodes, node * node_pc) { 
    for (int n=0; n<num_pc; n++) {
        for (int it = 0; it < num_nodes; ++it) {
            PRECISION voltage = node_pc[n].voltage[it];
            PRECISION charge = node_pc[n].charge[it];
            PRECISION capacitance = node_pc[n].capacitance[it];
            PRECISION leakage = node_pc[n].leakage[it];
            voltage += charge / capacitance;
            voltage *= (1.f - leakage);
            node_pc[n].voltage[it] = voltage;
            node_pc[n].charge[it]  = 0.f;
#ifdef _DEBUG
            printf("\t**node info **\n");
            printf("\tvoltage: %f, charge: %f\n", node_pc[n].voltage[it], node_pc[n].charge[it]);
#endif
        }//for: iterate nodess_per_piece
    }// for: pieces, DC
}

void getConfig(int argc, char ** argv){

    if (argc == 1){
        cout << "\n==== HELP ====\n-h or --help\tfor help\n-l or --loop\tto set loop times\n"
            "-p or --pieces\tto set pieces\n-n or --nodes\tto specify number of nodes\n"
            "-w or --wires\tto specify number of wires\n"
            "-c or --percent\tto specify pencentage of private nodes\n"
            "-s or --seed\tto specify random seed\n"
            "-b or --block\tto specify number of block\n"
            "-pe or --proc\tto specify number of process\n"
            "-t or --thread\tto speicify number of thread\n\n";
        exit(-1);
    }

    for (int i = 1; i < argc; i++){
        if ( !strcmp(argv[i], HELP_S) || !strcmp(argv[i], HELP_L) ) {

            cout << "\n==== HELP ====\n-h or --help\tfor help\n-l or --loop\tto set loop times\n"
                "-p or --pieces\tto set pieces\n-n or --nodes\tto specify number of nodes\n"
                "-w or --wires\tto specify number of wires\n"
                "-c or --percent\tto specify pencentage of private nodes\n"
                "-s or --seed\tto specify random seed\n"
                "-b or --block\tto specify number of block\n"
                "-pe or --proc\tto specify number of process\n"
                "-t or --thread\tto speicify number of thread\n\n";
            exit(-1);
        }
        else if ( !strcmp(argv[i], LOOP_S) || !strcmp(argv[i], LOOP_L) ) {
            num_loops = atoi(argv[i + 1]);
            i++;
        }
        else if ( !strcmp(argv[i], PIECE_S) || !strcmp(argv[i], PIECE_L) ) {
            num_pieces = atoi(argv[i + 1]);
            i++;
        }
        else if ( !strcmp(argv[i], NODE_S) || !strcmp(argv[i], NODE_L) ) {
            nodes_per_piece = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], WIRE_S) || !strcmp(argv[i], WIRE_L)){
            wires_per_piece = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], PERCENT_S) || !strcmp(argv[i], PERCENT_L)){
            pct_wire_in_piece = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], SEED_S) || !strcmp(argv[i], SEED_L)){
            random_seed = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], BLOCK_S) || !strcmp(argv[i], BLOCK_L)){
            num_blocks = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], THREAD_S) || !strcmp(argv[i], THREAD_L)){
            num_threads = atoi(argv[i + 1]);
            i++;
        }
        else if (!strcmp(argv[i], PE_S) || !strcmp(argv[i], PE_L)){
            num_pe = atoi(argv[i + 1]);
            i++;
        }
        else {
            cout << "Unknow parameter!" << endl;
            exit(-1);
        }
    }
}

int main(int argc, char ** argv) {
  
/* parameter setting */
    num_loops         = 1;
    num_pieces        = 4;
    nodes_per_piece   = 2;
    wires_per_piece   = 4;
    pct_wire_in_piece = 95;
    random_seed       = 0;
    num_blocks        = 32;
    num_threads       = 256;
    
    getConfig(argc, argv);

/* MPI init */
    int rank, commSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize); 
    num_pe = commSize;
    if (rank == 0) {
        printf("===========\nnum_loops:%d\nnum_pieces:%d\nnodes_per_piece:%d\nwires_per_piece:%d\n==========\n", num_loops, num_pieces, nodes_per_piece, wires_per_piece);
    }

    // set random seed
    srand48(random_seed);

//    int random_seed       = 12345;
    int steps             = STEPS;

    long num_circuit_nodes = num_pieces * nodes_per_piece;
    long num_circuit_wires = num_pieces * wires_per_piece;

    // calculate currents
    long operations = num_circuit_wires * (WIRE_SEGMENTS*6 + (WIRE_SEGMENTS-1)*4) * steps;
    // distribute charge
    operations += (num_circuit_wires * 4);
    // update voltages
    operations += (num_circuit_nodes * 4);
    // multiply by the number of loops
    operations *= num_loops;

/* worksplit */
    int pieces_per_pe = num_pieces / (num_pe-1);

/* circuit graph building */
    // node definition 
    node * node_piece;
    node * first_nodes;
    wire * wire_piece;
    wire * first_wires;
    if (rank == 0) {
        printf("Kernel Config: blocks:%d, threads; %d\n", num_blocks, num_threads);
        node_piece  = new node[num_pieces];
        first_nodes = new node[num_pieces];
        vector<int> shared_nodes_piece(num_pieces, 0); 
        // wire definition
        wire_piece = new wire[num_pieces];
        first_wires = new wire[num_pieces];
        // node initialization
        for (int n = 0; n < num_pieces; n++) {
            // allocate space for array in soa
            node_piece[n].capacitance = new PRECISION[nodes_per_piece];
            node_piece[n].leakage     = new PRECISION[nodes_per_piece];
            node_piece[n].charge      = new PRECISION[nodes_per_piece];
            node_piece[n].voltage     = new PRECISION[nodes_per_piece];
            node_piece[n].shr_pc      = new int[nodes_per_piece];
            node_piece[n].node_attr   = new int[nodes_per_piece];
            // initialize node_per_piece
            for (int i = 0; i < nodes_per_piece; i++) {
                // initialize node parameter
                node_piece[n].capacitance[i] = drand48() + 1.f;
                node_piece[n].leakage[i]     = 0.1f * drand48();
                node_piece[n].charge[i]      = 0.f;
                node_piece[n].voltage[i]     = 2*drand48() - 1.f;
                // node_attr (0:private, 1:shared)
                node_piece[n].shr_pc[i]      = 0;
                node_piece[n].node_attr[i]   = 0;
            
                // set first node in each piece
                if (i == 0) {
                    // allocate space for first node
                    // initialize first node
                    node_init(node_piece[n], i, first_nodes, n, 1);
                } //if
            }//for
        }//for

        // wire initialization
        for (int n = 0; n < num_pieces; n++) {
#ifdef _DEBUG
            printf("=== List nodes in piece %d ===\n", n);
#endif
            // allocate space for array in soa of wire
            wire_piece[n].currents    = new PRECISION*[wires_per_piece];
            for (int j=0; j<wires_per_piece; j++)
                wire_piece[n].currents[j]  = new PRECISION[WIRE_SEGMENTS];
            wire_piece[n].voltages    = new PRECISION*[wires_per_piece];
            for (int j=0; j<wires_per_piece; j++)
                wire_piece[n].voltages[j]  = new PRECISION[WIRE_SEGMENTS-1];
            wire_piece[n].resistance  = new PRECISION[wires_per_piece];
            wire_piece[n].inductance  = new PRECISION[wires_per_piece];
            wire_piece[n].capacitance = new PRECISION[wires_per_piece];
            wire_piece[n].in_ptr      = new int[wires_per_piece];
            wire_piece[n].out_ptr     = new int[wires_per_piece];
            wire_piece[n].wire_attr   = new int[wires_per_piece];
            // init wire shared part
            wire_piece[n].shr_voltage = new PRECISION[wires_per_piece];
            wire_piece[n].shr_charge  = new PRECISION[wires_per_piece];
            wire_piece[n].shr_pc      = new int[wires_per_piece];
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
                // init resistance
                wire_piece[n].resistance[i]  = drand48() * 10.0 + 1.0;
                // Keep inductance on the order of 1e-3 * dt to avoid resonance problems
                wire_piece[n].inductance[i]  = (drand48() + 0.1) * DELTAT * 1e-3;
                wire_piece[n].capacitance[i] = drand48() * 0.1;
                // UNC init connection
                wire_piece[n].in_ptr[i] = random_element(nodes_per_piece);
                //node_init(node_piece[n], random_element(nodes_per_piece), wire_piece[n].in_ptr, i, 1);
//            wire_piece[n].in_ptr[i][0] = random_element(nodes_per_piece);
                if ((100 * drand48()) < pct_wire_in_piece) {
                    wire_piece[n].out_ptr[i] = random_element(nodes_per_piece);
                //node_init(node_piece[n], random_element(nodes_per_piece), wire_piece[n].out_ptr, i, 1);
//                wire_piece[n].back().out_ptr = random_element(nodes_per_piece);
                }//if
                else {
#ifdef _DEBUG
                    cout << "\t\tShared appear\n";
#endif
                    // make wire as shared
                    wire_piece[n].wire_attr[i] = 1;
                    //node_piece[n].node_attr[wire_piece[n].in_ptr[i]] = 1;
                    //*((wire_piece[n].in_ptr[i]).node_attr) = 1;
                    // pick a random other piece and a node from there
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
                // Record the first wire pointer for this piece
                if (i == 0)
                    wire_init(wire_piece[n], i, first_wires, n, 1);
#ifdef _DEBUG
                // circuit info
                printf( "Wire %d resistance: %f, inductance: %f, capacitance: %f\n", i, wire_piece[n].resistance[i], wire_piece[n].inductance[i], wire_piece[n].capacitance[i]);
                printf("** node info **\n");
                printf("in_ptr/node_type:%d, capacitance: %f\n", node_piece[n].node_attr[(wire_piece[n].in_ptr[i])], node_piece[n].capacitance[(wire_piece[n].in_ptr[i])]);
                printf("out_ptr/node_type:%d, capacitance: %f\n", node_piece[n].node_attr[(wire_piece[n].out_ptr[i])], node_piece[n].capacitance[(wire_piece[n].out_ptr[i])]);
#endif
            }//for: wire_per_piece
        }//for : pieces
    }
    else {
        node_piece  = new node[pieces_per_pe];
        first_nodes = new node[pieces_per_pe];
        // wire definition
        wire_piece = new wire[pieces_per_pe];
        first_wires = new wire[pieces_per_pe];
        // node initialization
        for (int n = 0; n < pieces_per_pe; n++) {
            // allocate space for array in soa
            node_piece[n].capacitance = new PRECISION[nodes_per_piece];
            node_piece[n].leakage     = new PRECISION[nodes_per_piece];
            node_piece[n].charge      = new PRECISION[nodes_per_piece];
            node_piece[n].voltage     = new PRECISION[nodes_per_piece];
            node_piece[n].shr_pc      = new int[nodes_per_piece];
            node_piece[n].node_attr   = new int[nodes_per_piece];
        }
        // wire initialization
        for (int n = 0; n < pieces_per_pe; n++) {
            // allocate space for array in soa of wire
            wire_piece[n].currents    = new PRECISION*[wires_per_piece];
            for (int j=0; j<wires_per_piece; j++)
                wire_piece[n].currents[j]  = new PRECISION[WIRE_SEGMENTS];
            wire_piece[n].voltages    = new PRECISION*[wires_per_piece];
            for (int j=0; j<wires_per_piece; j++)
                wire_piece[n].voltages[j]  = new PRECISION[WIRE_SEGMENTS-1];
            wire_piece[n].resistance  = new PRECISION[wires_per_piece];
            wire_piece[n].inductance  = new PRECISION[wires_per_piece];
            wire_piece[n].capacitance = new PRECISION[wires_per_piece];
            wire_piece[n].in_ptr      = new int[wires_per_piece];
            wire_piece[n].out_ptr     = new int[wires_per_piece];
            wire_piece[n].wire_attr   = new int[wires_per_piece];
            // init wire shared part
            wire_piece[n].shr_voltage = new PRECISION[wires_per_piece];
            wire_piece[n].shr_charge  = new PRECISION[wires_per_piece];
            wire_piece[n].shr_pc      = new int[wires_per_piece];
        }
    }
    // global synchronization
    MPI_Barrier(MPI_COMM_WORLD);
    timeval t_b, t_e, t_kb, t_ke;
    double  t_p, t_kp;
    /* ditribute data to PEs */
    gettimeofday(&t_b, NULL);
    if (rank == 0) {
        for (int p=1; p<num_pe; p++) {
            for (int n=0; n<pieces_per_pe; n++) {
                int noffset = (p-1)*pieces_per_pe + n;
                // send node info
                MPI_Send(node_piece[noffset].capacitance, nodes_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                MPI_Send(node_piece[noffset].leakage    , nodes_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                MPI_Send(node_piece[noffset].charge     , nodes_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                MPI_Send(node_piece[noffset].voltage    , nodes_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                MPI_Send(node_piece[noffset].shr_pc     , nodes_per_piece, MPI_INT      , p, 0, MPI_COMM_WORLD);
                MPI_Send(node_piece[noffset].node_attr  , nodes_per_piece, MPI_INT      , p, 0, MPI_COMM_WORLD);
                // send wire info
                for (int i=0; i<wires_per_piece; i++)
                    MPI_Send(wire_piece[noffset].currents[i], WIRE_SEGMENTS  , MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                for (int i=0; i<wires_per_piece; i++)
                    MPI_Send(wire_piece[noffset].voltages[i] , WIRE_SEGMENTS-1, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                MPI_Send(wire_piece[noffset].resistance , wires_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                MPI_Send(wire_piece[noffset].inductance , wires_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                MPI_Send(wire_piece[noffset].capacitance, wires_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                MPI_Send(wire_piece[noffset].in_ptr     , wires_per_piece, MPI_INT      , p, 0, MPI_COMM_WORLD);
                MPI_Send(wire_piece[noffset].out_ptr    , wires_per_piece, MPI_INT      , p, 0, MPI_COMM_WORLD);
                MPI_Send(wire_piece[noffset].wire_attr  , wires_per_piece, MPI_INT      , p, 0, MPI_COMM_WORLD);
                MPI_Send(wire_piece[noffset].shr_voltage, wires_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                MPI_Send(wire_piece[noffset].shr_charge , wires_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                MPI_Send(wire_piece[noffset].shr_pc     , wires_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
            }
            
        }  
        
    }
    else {
        for (int n = 0; n < pieces_per_pe; n++) {
            // allocate space for array in soa
            MPI_Recv(node_piece[n].capacitance, nodes_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(node_piece[n].leakage    , nodes_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(node_piece[n].charge     , nodes_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(node_piece[n].voltage    , nodes_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(node_piece[n].shr_pc     , nodes_per_piece, MPI_INT      , 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(node_piece[n].node_attr  , nodes_per_piece, MPI_INT      , 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i=0; i<wires_per_piece; i++)
                MPI_Recv(wire_piece[n].currents[i], WIRE_SEGMENTS  , MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i=0; i<wires_per_piece; i++)
                MPI_Recv(wire_piece[n].voltages[i] , WIRE_SEGMENTS-1, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(wire_piece[n].resistance , wires_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(wire_piece[n].inductance , wires_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(wire_piece[n].capacitance, wires_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(wire_piece[n].in_ptr     , wires_per_piece, MPI_INT      , 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(wire_piece[n].out_ptr    , wires_per_piece, MPI_INT      , 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(wire_piece[n].wire_attr  , wires_per_piece, MPI_INT      , 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(wire_piece[n].shr_voltage, wires_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(wire_piece[n].shr_charge , wires_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(wire_piece[n].shr_pc     , wires_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }
    }
    /* Computing circuit graph */
/*
    // CPU: main loop 
    for (int iloop = 0; iloop < num_loops; iloop++) { 
        // calculate currents cpu
        calculate_current_cpu(num_pieces, wires_per_piece, wire_piece, node_piece);
        // distributed charge cpu
        distributed_charge_cpu(num_pieces, wires_per_piece, wire_piece, node_piece);
        // update voltage cpu
        update_voltage_cpu(num_pieces, nodes_per_piece, node_piece);
    }// for: mainloop
*/
    // GPU: main loop
    // GPU initialization
    PRECISION * d_node_capacitance, * d_node_leakage, * d_node_charge, * d_node_voltage;
    PRECISION * d_wire_currents, * d_wire_voltages, * d_wire_resistance, * d_wire_inductance, * d_wire_capacitance;
    PRECISION * d_shr_voltage, * d_shr_charge;
    int       * d_in_ptr, * d_out_ptr, * d_shr_pc, * d_wire_attr;
    int         deviceCnt = 0;
    // GPU setDeivce
    cudaCheckError( __LINE__, cudaGetDeviceCount(&deviceCnt));
    cudaCheckError( __LINE__, cudaSetDevice(rank % deviceCnt));
    if (rank) {
        // GPU allocation
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_capacitance, sizeof(PRECISION)*nodes_per_piece));
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_leakage    , sizeof(PRECISION)*nodes_per_piece));
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_charge     , sizeof(PRECISION)*nodes_per_piece));
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_voltage    , sizeof(PRECISION)*nodes_per_piece));
        
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_currents   , sizeof(PRECISION)*wires_per_piece*WIRE_SEGMENTS));  
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_voltages   , sizeof(PRECISION)*wires_per_piece*(WIRE_SEGMENTS-1)));   
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_resistance , sizeof(PRECISION)*wires_per_piece));   
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_inductance , sizeof(PRECISION)*wires_per_piece));
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_capacitance, sizeof(PRECISION)*wires_per_piece));
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_in_ptr, sizeof(int)*wires_per_piece));    
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_out_ptr, sizeof(int)*wires_per_piece));    
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_shr_voltage     , sizeof(PRECISION)*wires_per_piece));
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_shr_charge      , sizeof(PRECISION)*wires_per_piece));
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_shr_pc, sizeof(int)*wires_per_piece));
        cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_attr       , sizeof(int)*wires_per_piece));    
    } // end if rank   
    /* GPU main loop */
    for (int iloop = 0; iloop < num_loops; iloop++) { 
   
        if (rank) { 
    /* computation: calculate currents & distributed charge */
            for (int n=0; n<pieces_per_pe; n++) {
                // CPU to GPU memcpy
                cudaCheckError( __LINE__, cudaMemcpy( d_node_capacitance, node_piece[n].capacitance, sizeof(PRECISION)*nodes_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_node_leakage    , node_piece[n].leakage    , sizeof(PRECISION)*nodes_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_node_charge     , node_piece[n].charge     , sizeof(PRECISION)*nodes_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_node_voltage    , node_piece[n].voltage    , sizeof(PRECISION)*nodes_per_piece, cudaMemcpyHostToDevice));

                for (int i = 0; i < wires_per_piece; i++) {
                    int coffset = i * WIRE_SEGMENTS;
                    int voffset = i * (WIRE_SEGMENTS-1);
                    cudaCheckError( __LINE__, cudaMemcpy( (d_wire_currents+coffset) , wire_piece[n].currents[i] , sizeof(PRECISION)*WIRE_SEGMENTS, cudaMemcpyHostToDevice));
                    cudaCheckError( __LINE__, cudaMemcpy( (d_wire_voltages+voffset) , wire_piece[n].voltages[i] , sizeof(PRECISION)*(WIRE_SEGMENTS-1), cudaMemcpyHostToDevice));
                } 
                cudaCheckError( __LINE__, cudaMemcpy( d_wire_resistance , wire_piece[n].resistance , sizeof(PRECISION)*wires_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_wire_inductance , wire_piece[n].inductance , sizeof(PRECISION)*wires_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_wire_capacitance, wire_piece[n].capacitance, sizeof(PRECISION)*wires_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_in_ptr          , wire_piece[n].in_ptr     , sizeof(int)*wires_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_out_ptr         , wire_piece[n].out_ptr    , sizeof(int)*wires_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_shr_voltage     , wire_piece[n].shr_voltage, sizeof(PRECISION)*wires_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_shr_charge      , wire_piece[n].shr_charge , sizeof(PRECISION)*wires_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_shr_pc          , wire_piece[n].shr_pc     , sizeof(int)*wires_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_wire_attr       , wire_piece[n].wire_attr  , sizeof(int)*wires_per_piece, cudaMemcpyHostToDevice));

                // <<<calculate currents>>> gpu
                gettimeofday(&t_kb, NULL);
                calculate_current_gpu<<<num_blocks, num_threads>>>(wires_per_piece, d_wire_currents, d_wire_voltages, d_in_ptr, d_out_ptr, d_wire_inductance, d_wire_resistance, d_wire_capacitance, d_node_voltage, d_wire_attr, d_shr_voltage);
                cudaCheckError( __LINE__, cudaDeviceSynchronize()); 
                gettimeofday(&t_ke, NULL);
                t_kp = (t_ke.tv_sec + t_ke.tv_usec * 1e-6) - (t_kb.tv_sec + t_kb.tv_usec * 1e-6);
                // <<<distributed charge>>> gpu
                gettimeofday(&t_kb, NULL);
                distributed_charge_gpu<<<num_blocks, num_threads>>>(wires_per_piece, d_wire_currents,  d_in_ptr, d_out_ptr, d_node_charge, d_wire_attr, d_shr_charge);
                cudaCheckError( __LINE__, cudaDeviceSynchronize()); 
                gettimeofday(&t_ke, NULL);
                t_kp += (t_ke.tv_sec + t_ke.tv_usec * 1e-6) - (t_kb.tv_sec + t_kb.tv_usec * 1e-6);

                // GPU to CPU memcpy
                cudaCheckError( __LINE__, cudaMemcpy( node_piece[n].charge, d_node_charge     ,  sizeof(PRECISION)*nodes_per_piece, cudaMemcpyDeviceToHost));
                cudaCheckError( __LINE__, cudaMemcpy( wire_piece[n].shr_charge, d_shr_charge      ,  sizeof(PRECISION)*wires_per_piece, cudaMemcpyDeviceToHost));
                for (int i = 0; i < wires_per_piece; i++) {
                    int coffset = i * WIRE_SEGMENTS;
                    int voffset = i * (WIRE_SEGMENTS-1);
                    cudaCheckError( __LINE__, cudaMemcpy( wire_piece[n].currents[i], (d_wire_currents+coffset) ,  sizeof(PRECISION)*WIRE_SEGMENTS, cudaMemcpyDeviceToHost));
                    cudaCheckError( __LINE__, cudaMemcpy( wire_piece[n].voltages[i], (d_wire_voltages+voffset) ,  sizeof(PRECISION)*(WIRE_SEGMENTS-1), cudaMemcpyDeviceToHost));
                }// for wire_per_piece
            } // for: piece_gpu
    /* post work for charge distribution */
            // computing PE send post_work to main PE
            for (int i=0; i<pieces_per_pe; i++) {
                MPI_Send(node_piece[i].charge, nodes_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD);
            }
            for (int i=0; i<pieces_per_pe; i++) {
                MPI_Send(wire_piece[i].shr_charge , wires_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD);
            }
            // update post_work from main PE
            for (int i=0; i<pieces_per_pe; i++) {
                MPI_Recv(node_piece[i].charge, nodes_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int i=0; i<pieces_per_pe; i++) {
                for (int j=0; j<wires_per_piece; j++) {
                    wire_piece[i].shr_charge[j] = 0.f;    
                }
            }
        } else { // endif rank
            // receive post_work from main PE
            for (int p=1; p<num_pe; p++) {
                for (int n=0; n<pieces_per_pe; n++) {
                    int noffset = (p-1)*pieces_per_pe + n;
                    MPI_Recv(node_piece[noffset].charge, nodes_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }        
            for (int p=1; p<num_pe; p++) {
                for (int n=0; n<pieces_per_pe; n++) {
                    int noffset = (p-1)*pieces_per_pe + n;
                    MPI_Recv(wire_piece[noffset].shr_charge, wires_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
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
            // send post_work to PEs
            for (int p=1; p<num_pe; p++) {
                for (int n=0; n<pieces_per_pe; n++) {
                    int noffset = (p-1)*pieces_per_pe + n;
                    MPI_Send(node_piece[noffset].charge, nodes_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD);
                }
            }        
#ifdef _DEBUG
            for (int n=0; n<num_pieces; n++) {
                for (int it = 0; it<wires_per_piece; ++it) {
                    printf("\t**node info **\n");
                    printf("\tin_charge: %f, out_charge: %f\n", node_piece[n].charge[wire_piece[n].in_ptr[it]], node_piece[n].charge[wire_piece[n].out_ptr[it]]);
                }
            }
            printf("++++++++++++++++++++++++++++++++++++++++++++++++++\n");
#endif

        } // else end rank 0
    /* computation: update voltage */
        if (rank) {
            for (int n=0; n<pieces_per_pe; n++) {
                // CPU to GPU memcpy
                cudaCheckError( __LINE__, cudaMemcpy( d_node_capacitance, node_piece[n].capacitance, sizeof(PRECISION)*nodes_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_node_leakage    , node_piece[n].leakage    , sizeof(PRECISION)*nodes_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_node_charge     , node_piece[n].charge     , sizeof(PRECISION)*nodes_per_piece, cudaMemcpyHostToDevice));
                cudaCheckError( __LINE__, cudaMemcpy( d_node_voltage    , node_piece[n].voltage    , sizeof(PRECISION)*nodes_per_piece, cudaMemcpyHostToDevice));
                // update voltage gpu
                gettimeofday(&t_kb, NULL);
                update_voltage_gpu<<<num_blocks, num_threads>>>(nodes_per_piece, d_node_voltage, d_node_charge, d_node_capacitance, d_node_leakage);
                cudaCheckError( __LINE__, cudaDeviceSynchronize()); 
                gettimeofday(&t_ke, NULL);
                t_kp += (t_ke.tv_sec + t_ke.tv_usec * 1e-6) - (t_kb.tv_sec + t_kb.tv_usec * 1e-6);
                // GPU to CPU memcpy
                cudaCheckError( __LINE__, cudaMemcpy( node_piece[n].charge, d_node_charge, sizeof(PRECISION)*nodes_per_piece, cudaMemcpyDeviceToHost));
                cudaCheckError( __LINE__, cudaMemcpy( node_piece[n].voltage, d_node_voltage, sizeof(PRECISION)*nodes_per_piece, cudaMemcpyDeviceToHost));
            }
    /* post_work for updating voltage */
            // computing PE send post_work to main PE
            for (int i=0; i<pieces_per_pe; i++) {
                MPI_Send(node_piece[i].voltage, nodes_per_piece, MPI_PRECISION, 0, 0, MPI_COMM_WORLD);
            }
            
        } else {//endif rank 
            // receive post_work from main PE
            for (int p=1; p<num_pe; p++) {
                for (int n=0; n<pieces_per_pe; n++) {
                    int noffset = (p-1)*pieces_per_pe + n;
                    MPI_Recv(node_piece[noffset].voltage, nodes_per_piece, MPI_PRECISION, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }        
            // post work to update shared voltage
            for (int n=0; n<num_pieces; n++) {
                for (int i=0; i<wires_per_piece; i++) {
                    if (wire_piece[n].wire_attr[i] == 1) {
                       wire_piece[n].shr_voltage[i] =  node_piece[wire_piece[n].shr_pc[i]].voltage[wire_piece[n].out_ptr[i]];
                    }
                }
            }
    /* result showing for GPU */
#ifdef _DEBUG
            for (int n=0; n<num_pieces; n++) {
                for (int it=0; it<nodes_per_piece; it++) {
                    printf("\t**node info **\n");
                    printf("\tvoltage: %f, charge: %f\n", node_piece[n].voltage[it], node_piece[n].charge[it]);
                }
            }
#endif
        }// endelse rank0
    }// main loop end
    gettimeofday(&t_e, NULL);
    // result info
    char hostname[256];
    gethostname(hostname, 256);
    t_p = (t_e.tv_sec + t_e.tv_usec * 1e-6) - (t_b.tv_sec + t_b.tv_usec * 1e-6);
    printf("Rank:%d, on host: %s, device_id: %d, kernel time: %f\n", rank, hostname, rank % deviceCnt, t_kp);
    printf("Time: %f\n", t_p);
    /* free cudamem */
    if (rank) {
        // GPU allocation
        cudaCheckError( __LINE__, cudaFree(d_node_capacitance));    
        cudaCheckError( __LINE__, cudaFree(d_node_leakage));    
        cudaCheckError( __LINE__, cudaFree(d_node_charge));    
        cudaCheckError( __LINE__, cudaFree(d_node_voltage));    
        cudaCheckError( __LINE__, cudaFree(d_wire_currents)); 
        cudaCheckError( __LINE__, cudaFree(d_wire_voltages));   
        cudaCheckError( __LINE__, cudaFree(d_wire_resistance));   
        cudaCheckError( __LINE__, cudaFree(d_wire_inductance));
        cudaCheckError( __LINE__, cudaFree(d_wire_capacitance));    
        cudaCheckError( __LINE__, cudaFree(d_in_ptr));    
        cudaCheckError( __LINE__, cudaFree(d_out_ptr));    
        cudaCheckError( __LINE__, cudaFree(d_shr_voltage));    
        cudaCheckError( __LINE__, cudaFree(d_shr_charge));    
        cudaCheckError( __LINE__, cudaFree(d_shr_pc));    
        cudaCheckError( __LINE__, cudaFree(d_wire_attr));    
    }
    MPI_Finalize();
    return 0;
}


























































