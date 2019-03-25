#include "circuit.h"
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>

using namespace std;

inline void cudaCheckError(int line, cudaError_t ce)
{
    if (ce != cudaSuccess){
        printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
        exit(1);
    }
}

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
        }//for: iterate wires_per_pc
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

void cudaRun(node * node_pc, wire * wire_pc, unsigned char * transfer_buf, int nodes_per_pc, int wires_per_pc, int pieces_per_pe, int peid, int num_blocks, int num_threads) {
    
    // GPU initialization
    PRECISION * d_node_capacitance, * d_node_leakage, * d_node_charge, * d_node_voltage;
    PRECISION * d_wire_currents, * d_wire_voltages, * d_wire_resistance, * d_wire_inductance, * d_wire_capacitance;
    PRECISION * d_shr_voltage, * d_shr_charge;
    int       * d_in_ptr, * d_out_ptr, * d_shr_pc, * d_wire_attr;
#if 1
        for (int n=0; n<pieces_per_pe; n++) {
            for (int it=0; it<nodes_per_pc; it++) {
                printf("\t**node info **\n");
                printf("\tvoltage: %f, charge: %f\n", node_pc[n].voltage[it], node_pc[n].charge[it]);
            }
        }
#endif
#if 0
            for (int n = 0; n < pieces_per_pe; n++) {
                for (int i = 0; i < wires_per_pc; i++) {

                   // circuit info
                   printf( "Wire %d resistance: %f, inductance: %f, capacitance: %f\n", i, wire_pc[n].resistance[i], wire_pc[n].inductance[i], wire_pc[n].capacitance[i]);
                   printf("** node info **\n");
                   printf("in_ptr/node_type:%d, capacitance: %f\n", node_pc[n].node_attr[(wire_pc[n].in_ptr[i])], node_pc[n].capacitance[(wire_pc[n].in_ptr[i])]);
                   printf("out_ptr/node_type:%d, capacitance: %f\n", node_pc[n].node_attr[(wire_pc[n].out_ptr[i])], node_pc[n].capacitance[(wire_pc[n].out_ptr[i])]);
                }
            } 
#endif
    // GPU allocation
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_capacitance, sizeof(PRECISION)*nodes_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_leakage    , sizeof(PRECISION)*nodes_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_charge     , sizeof(PRECISION)*nodes_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_voltage    , sizeof(PRECISION)*nodes_per_pc));

    cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_currents   , sizeof(PRECISION)*wires_per_pc*WIRE_SEGMENTS));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_voltages   , sizeof(PRECISION)*wires_per_pc*(WIRE_SEGMENTS-1)));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_resistance , sizeof(PRECISION)*wires_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_inductance , sizeof(PRECISION)*wires_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_capacitance, sizeof(PRECISION)*wires_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_in_ptr, sizeof(int)*wires_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_out_ptr, sizeof(int)*wires_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_shr_voltage     , sizeof(PRECISION)*wires_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_shr_charge      , sizeof(PRECISION)*wires_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_shr_pc, sizeof(int)*wires_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_wire_attr       , sizeof(int)*wires_per_pc));
    /* computation: calculate currents & distributed charge */
    for (int n=0; n<pieces_per_pe; n++) {
        // CPU to GPU memcpy
        cudaCheckError( __LINE__, cudaMemcpy( d_node_capacitance, node_pc[n].capacitance, sizeof(PRECISION)*nodes_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_node_leakage    , node_pc[n].leakage    , sizeof(PRECISION)*nodes_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_node_charge     , node_pc[n].charge     , sizeof(PRECISION)*nodes_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_node_voltage    , node_pc[n].voltage    , sizeof(PRECISION)*nodes_per_pc, cudaMemcpyHostToDevice));

        for (int i = 0; i < wires_per_pc; i++) {
            int coffset = i * WIRE_SEGMENTS;
            int voffset = i * (WIRE_SEGMENTS-1);
            cudaCheckError( __LINE__, cudaMemcpy( (d_wire_currents+coffset) , wire_pc[n].currents[i] , sizeof(PRECISION)*WIRE_SEGMENTS, cudaMemcpyHostToDevice));
            cudaCheckError( __LINE__, cudaMemcpy( (d_wire_voltages+voffset) , wire_pc[n].voltages[i] , sizeof(PRECISION)*(WIRE_SEGMENTS-1), cudaMemcpyHostToDevice));
        }
        cudaCheckError( __LINE__, cudaMemcpy( d_wire_resistance , wire_pc[n].resistance , sizeof(PRECISION)*wires_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_wire_inductance , wire_pc[n].inductance , sizeof(PRECISION)*wires_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_wire_capacitance, wire_pc[n].capacitance, sizeof(PRECISION)*wires_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_in_ptr          , wire_pc[n].in_ptr     , sizeof(int)*wires_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_out_ptr         , wire_pc[n].out_ptr    , sizeof(int)*wires_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_shr_voltage     , wire_pc[n].shr_voltage, sizeof(PRECISION)*wires_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_shr_charge      , wire_pc[n].shr_charge , sizeof(PRECISION)*wires_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_shr_pc          , wire_pc[n].shr_pc     , sizeof(int)*wires_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_wire_attr       , wire_pc[n].wire_attr  , sizeof(int)*wires_per_pc, cudaMemcpyHostToDevice));
        // <<<calculate currents>>> gpu
        calculate_current_gpu<<<num_blocks, num_threads>>>(wires_per_pc, d_wire_currents, d_wire_voltages, d_in_ptr, d_out_ptr, d_wire_inductance, d_wire_resistance, d_wire_capacitance, d_node_voltage, d_wire_attr, d_shr_voltage);
        cudaCheckError( __LINE__, cudaDeviceSynchronize());
        // <<<distributed charge>>> gpu
        distributed_charge_gpu<<<num_blocks, num_threads>>>(wires_per_pc, d_wire_currents,  d_in_ptr, d_out_ptr, d_node_charge, d_wire_attr, d_shr_charge);
        cudaCheckError( __LINE__, cudaDeviceSynchronize());

        // GPU to CPU memcpy
        cudaCheckError( __LINE__, cudaMemcpy( node_pc[n].charge, d_node_charge     ,  sizeof(PRECISION)*nodes_per_pc, cudaMemcpyDeviceToHost));
        cudaCheckError( __LINE__, cudaMemcpy( wire_pc[n].shr_charge, d_shr_charge      ,  sizeof(PRECISION)*wires_per_pc, cudaMemcpyDeviceToHost));
        for (int i = 0; i < wires_per_pc; i++) {
            int coffset = i * WIRE_SEGMENTS;
            int voffset = i * (WIRE_SEGMENTS-1);
            cudaCheckError( __LINE__, cudaMemcpy( wire_pc[n].currents[i], (d_wire_currents+coffset) ,  sizeof(PRECISION)*WIRE_SEGMENTS, cudaMemcpyDeviceToHost));
            cudaCheckError( __LINE__, cudaMemcpy( wire_pc[n].voltages[i], (d_wire_voltages+voffset) ,  sizeof(PRECISION)*(WIRE_SEGMENTS-1), cudaMemcpyDeviceToHost));
        }// for wire_per_piece
    }
#if 0
            for (int n = 0; n < pieces_per_pe; n++) {
                for (int i = 0; i < wires_per_pc; i++) {

                   // circuit info
                   printf( "Wire %d resistance: %f, inductance: %f, capacitance: %f\n", i, wire_pc[n].resistance[i], wire_pc[n].inductance[i], wire_pc[n].capacitance[i]);
                }
            } 
#endif
#if 0
        for (int n=0; n<pieces_per_pe; n++) {
            for (int it=0; it<nodes_per_pc; it++) {
                printf("\t**node info **\n");
                printf("\tvoltage: %f, charge: %f\n", node_pc[n].voltage[it], node_pc[n].charge[it]);
            }
        }
#endif
    /* post work for charge distribution */
    int post_size = sizeof(int) + pieces_per_pe * sizeof(PRECISION) * (nodes_per_pc + wires_per_pc);
    unsigned char * post_mem = transfer_buf;
    int * index_init = reinterpret_cast<int *>(post_mem);
    * index_init = peid;
    PRECISION * data_init = reinterpret_cast<PRECISION *>(post_mem+sizeof(int));
    // init transfer buffer
    for (int n=0; n<pieces_per_pe; n++) {
#if 0
        data_init = node_pc[n].charge;
        data_init += nodes_per_pc * sizeof(PRECISION);
        data_init = wire_pc[n].shr_charge;
        data_init += wires_per_pc * sizeof(PRECISION);
#else
        memcpy(data_init+n*(nodes_per_pc + wires_per_pc), node_pc[n].charge, nodes_per_pc*sizeof(PRECISION));
        memcpy(data_init+n*(wires_per_pc+nodes_per_pc) + nodes_per_pc, wire_pc[n].shr_charge, wires_per_pc*sizeof(PRECISION));
#endif
    }
#if 1
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
#endif
}
void cudaPost(node * node_pc, wire * wire_pc, unsigned char * result_buf, int nodes_per_pc, int wires_per_pc, int pieces_per_pe, int peid, int num_blocks, int num_threads) {

    PRECISION * d_node_capacitance, * d_node_leakage, * d_node_charge, * d_node_voltage;
    // GPU allocation
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_capacitance, sizeof(PRECISION)*nodes_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_leakage    , sizeof(PRECISION)*nodes_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_charge     , sizeof(PRECISION)*nodes_per_pc));
    cudaCheckError( __LINE__, cudaMalloc((void **) &d_node_voltage    , sizeof(PRECISION)*nodes_per_pc));
#if 0
        for (int n=0; n<pieces_per_pe; n++) {
            for (int it=0; it<nodes_per_pc; it++) {
                printf("\t**node info **\n");
                printf("\tvoltage: %f, charge: %f\n", node_pc[n].voltage[it], node_pc[n].charge[it]);
            }
        }
#endif
#if 1
            for (int n = 0; n < pieces_per_pe; n++) {
                for (int i = 0; i < nodes_per_pc; i++) {

                   // circuit info
                   printf( "Node %d charge: %f, voltage: %f, capacitance: %f, leakage: %f\n", i, node_pc[n].charge[i], node_pc[n].voltage[i], node_pc[n].capacitance[i], node_pc[n].leakage[i]);
                }
            }
#endif
    for (int n=0; n<pieces_per_pe; n++) {
        // CPU to GPU memcpy
        cudaCheckError( __LINE__, cudaMemcpy( d_node_capacitance, node_pc[n].capacitance, sizeof(PRECISION)*nodes_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_node_leakage    , node_pc[n].leakage    , sizeof(PRECISION)*nodes_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_node_charge     , node_pc[n].charge     , sizeof(PRECISION)*nodes_per_pc, cudaMemcpyHostToDevice));
        cudaCheckError( __LINE__, cudaMemcpy( d_node_voltage    , node_pc[n].voltage    , sizeof(PRECISION)*nodes_per_pc, cudaMemcpyHostToDevice));
        // update voltage gpu
        update_voltage_gpu<<<num_blocks, num_threads>>>(nodes_per_pc, d_node_voltage, d_node_charge, d_node_capacitance, d_node_leakage);
        cudaCheckError( __LINE__, cudaDeviceSynchronize());
        // GPU to CPU memcpy
        cudaCheckError( __LINE__, cudaMemcpy( node_pc[n].charge, d_node_charge, sizeof(PRECISION)*nodes_per_pc, cudaMemcpyDeviceToHost));
        cudaCheckError( __LINE__, cudaMemcpy( node_pc[n].voltage, d_node_voltage, sizeof(PRECISION)*nodes_per_pc, cudaMemcpyDeviceToHost));
    }
#if 0
        for (int n=0; n<pieces_per_pe; n++) {
            for (int it=0; it<nodes_per_pc; it++) {
                printf("\t**node info **\n");
                printf("\tvoltage: %f, charge: %f\n", node_pc[n].voltage[it], node_pc[n].charge[it]);
            }
        }
#endif
    /* result work for charge distribution */
    int result_size = sizeof(int) + pieces_per_pe * sizeof(PRECISION) * nodes_per_pc * 2;
    unsigned char * result_mem = result_buf;
    int * index_init = reinterpret_cast<int *>(result_mem);
    * index_init = peid;
    result_mem += sizeof(int);
    PRECISION * data_init = reinterpret_cast<PRECISION *>(result_mem);
    // init transfer buffer
    for (int n=0; n<pieces_per_pe; n++) {
        memcpy(data_init + n*nodes_per_pc*2, node_pc[n].voltage, nodes_per_pc*sizeof(PRECISION));
        memcpy(data_init + n*nodes_per_pc*2 + nodes_per_pc, node_pc[n].charge , nodes_per_pc*sizeof(PRECISION));
    }
    // GPU allocation
    cudaCheckError( __LINE__, cudaFree(d_node_capacitance));
    cudaCheckError( __LINE__, cudaFree(d_node_leakage));
    cudaCheckError( __LINE__, cudaFree(d_node_charge));
    cudaCheckError( __LINE__, cudaFree(d_node_voltage));
}

