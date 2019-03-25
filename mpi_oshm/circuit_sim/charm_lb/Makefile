#CHARM_DIR = /home/ruidong/Documents/software/charm/examples/charm++/net-linux-x86_64
CHARM_DIR = /home/rgu/software/charm/mpi-linux-x86_64#/cluster/software/charm/charm-6.7.0-net#/usr/local/charm/netlrts-linux-x86_64-smp
CHARM_INC = -I$(CHARM_DIR)/include
CHARM_LIB = -L$(CHARM_DIR)/lib
CHARM_LB=-module CommonLBs
CC_FLAG   = -language charm++
OBJS = circuit.o circuitcuda.o

CUDA_DIR  = /usr/local/cuda
CUDA_INC  = -I$(CUDA_DIR)/include
CUDA_LIB  = -L$(CUDA_DIR)/lib64
NVCC_FLAG = -O2 -c -arch=sm_20
NVCC_LFLAG = -lcuda -lcudart

CHARMC = charmc
NVCC   = nvcc

BINARY = circuit_sim

all: circuit_sim
circuit_sim: $(OBJS)
	$(CHARMC) $(CHARM_LB) -language charm++ -g -o circuit_sim  $(OBJS) $(CHARM_LIB) $(CUDA_LIB) $(NVCC_LFLAG)
projections: $(COMPOBJS)
	$(CHARMC) -language charm++ -tracemode projections -g -lz -o circuit_sim.prj $(OBJS)

circuit.decl.h: circuit.ci
	$(CHARMC) -g circuit.ci
circuit.o: circuit.C circuit.decl.h circuit.h
	$(CHARMC) -g -O2 -c circuit.C
circuitcuda.o: circuitcuda.cu
	$(NVCC) $(NVCC_FLAG) $(CHARM_INC) -g -o circuitcuda.o circuitcuda.cu

clean:
	rm -f *.o *decl.h *def.h $(BINARY) charmrun
