##### Commands #####
CHARMC = charmc
NVCC   = nvcc

##### Compiler #####

##### LoadBalance setting #####
CHARM_LB=-module CommonLBs

##### Object #####
OBJS =nw.o nwcuda.o

##### Directry #####
NVCC_DIR =/usr/local/cuda

##### LIBINC #####
NVCC_LIB  = -L$(NVCC_DIR)/lib64
NVCC_INC  = -I$(NVCC_DIR)/include
CHARM_INC = -I/home/rgu/software/charm/mpi-linux-x86_64/include#-I/cluster/software/charm/charm-6.7.0-net/include #-I/share/sw/charm++/6.6.1/mpi-linux-x86_64

##### Flags #####
NVCC_FLAGS = -c --cudart=shared #--arch=sm_20 -device-debug -deviceemu
CPP_FLAGS  = -c -D_USE_STATIC_LIB -Wall -fPIC -g #-D_DOUBLE_PRECISION

all: nw

nw: $(OBJS)
	$(CHARMC) $(CHARM_LB) -language charm++ -g -o nw $(OBJS) $(NVCC_LIB) -lcuda -lcudart 

projections: $(COMPOBJS)
	$(CHARMC) -language charm++ -tracemode projections -g -lz -o nw.prj $(OBJS)


nw.decl.h: nw.ci
	$(CHARMC) -g nw.ci

clean:
	rm -f *.decl.h *.def.h *.o nw charmrun

nw.o: nw.C nw.decl.h
	$(CHARMC) -g -c nw.C

nwcuda.o: nwcuda.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $(CHARM_INC) -g -o nwcuda.o nwcuda.cu

test: all
	./charmrun ./nw +p4 8 8

