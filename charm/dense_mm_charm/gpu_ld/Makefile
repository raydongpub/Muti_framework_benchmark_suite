##### Commands #####
CHARMC = charmc
NVCC   = /usr/local/cuda/bin/nvcc

##### Compiler #####

##### LoadBalance setting #####
CHARM_LB=-module CommonLBs

##### Object #####
OBJS =matmul.o matmulcudaCU.o matdataset.o

##### Directry #####
NVCC_DIR =/usr/local/cuda

##### LIBINC #####
NVCC_LIB  = -L$(NVCC_DIR)/lib64
NVCC_INC  = -I$(NVCC_DIR)/include
CHARM_INC = -I/home/rgu/software/charm/mpi-linux-x86_64/include #/cluster/software/charm/charm-6.7.0-net/include#/share/sw/charm++/6.6.1/mpi-linux-x86_64

##### Flags #####
NVCC_FLAGS = -O2 -c #-device-debug -deviceemu
CPP_FLAGS  = -c -O2 -D_USE_STATIC_LIB -Wall -fPIC -g #-D_DOUBLE_PRECISION

all: matmul

matmul: $(OBJS)
	$(CHARMC) $(CHARM_LB) -language charm++ -g -o matmul $(OBJS) $(NVCC_LIB) -lcuda -lcudart 

projections: $(COMPOBJS)
	$(CHARMC) -language charm++ -tracemode projections -g -lz -o matmul.prj $(OBJS)


matmul.decl.h: matmul.ci
	$(CHARMC) -g matmul.ci

clean:
	rm -f *.decl.h *.def.h *.o matmul charmrun

matmul.o: matmul.C matmul.decl.h
	$(CHARMC) -g -O2 -c matmul.C

matmulcudaCU.o: matmulcuda.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $(CHARM_INC) -g -o matmulcudaCU.o matmulcuda.cu

matdataset.o : matdataset.cpp
	$(CHARMC) $(CPP_FLAGS) $< -o $@

test: all
	./charmrun ./matmul +p4 8 8

