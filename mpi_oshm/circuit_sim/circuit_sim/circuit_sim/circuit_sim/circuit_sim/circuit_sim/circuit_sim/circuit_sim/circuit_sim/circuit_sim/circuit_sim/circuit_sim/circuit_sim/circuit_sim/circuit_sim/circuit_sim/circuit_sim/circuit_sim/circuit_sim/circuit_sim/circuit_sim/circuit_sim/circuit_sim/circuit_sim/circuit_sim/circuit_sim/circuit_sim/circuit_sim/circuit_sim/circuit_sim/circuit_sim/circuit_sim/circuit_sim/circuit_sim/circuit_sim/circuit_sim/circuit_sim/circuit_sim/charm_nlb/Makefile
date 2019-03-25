#CHARM_DIR = /home/ruidong/Documents/software/charm/examples/charm++/net-linux-x86_64
CHARM_DIR = /usr/local/charm/mpi-linux-x86_64-smp
CHARM_INC = -I$(CHARM_DIR)/include
CHARM_LIB = -L$(CHARM_DIR)/lib
CC_FLAG   = -language charm++
OBJS = circuit.o circuitcuda.o

CUDA_DIR  = /usr/local/cuda
CUDA_INC  = -I$(CUDA_DIR)/include
CUDA_LIB  = -L$(CUDA_DIR)/lib64
NVCC_FLAG = -arch=sm_20
NVCC_LFLAG = -lcuda -lcudart

CHARMC = charmc
NVCC   = nvcc

BINARY = circuit_sim

all: $(BINARY)
$(BINARY): $(OBJS)
	$(CHARMC) $(CHARM_INC) $(CC_FLAG) -o $@ $^ $(CHARM_LIB) $(CUDA_LIB) $(NVCC_LFLAG)

circuit.o: circuit.C circuit.decl.h
	$(CHARMC) -c circuit.C
circuit.decl.h: circuit.ci
	$(CHARMC) $<
circuitcuda.o: circuitcuda.cu
	$(NVCC) $(NVCC_FLAG) -c $<

clean:
	rm -f *.o *decl.h *def.h $(BINARY) charmrun
