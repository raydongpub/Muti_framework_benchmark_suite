##### Commands #####
CHARMC = charmc
NVCC   = nvcc
CPP    = g++
##### Compiler #####

##### LoadBalance setting #####
CHARM_LB=-module CommonLBs

##### Object #####
OBJS =commonBMT.o himenocuda.o himeno.o 
##### Directry #####
NVCC_DIR =/usr/local/cuda

##### LIBINC #####
NVCC_LIB  = -L$(NVCC_DIR)/lib64
NVCC_INC  = -I$(NVCC_DIR)/include
CHARM_INC = -I/home/rgu/software/charm/mpi-linux-x86_64/include#-I/cluster/software/charm/charm-6.7.0-net/include#-I/home/ruidong/Documents/software/charm/net-linux-x86_64/include

##### Flags #####
NVCC_FLAGS = -arch=sm_20 -O2 -c #-D_DOUBLE_PRECISION#-device-debug -deviceemu
CPP_FLAGS  = -cpp-option -c -O2 -D_USE_STATIC_LIB -Wall -fPIC -g #-D_DOUBLE_PRECISION

all : himeno

himeno : $(OBJS)
	$(CHARMC) $(CHARM_LB) -language charm++ -o himeno $(OBJS) $(NVCC_LIB) -lcuda -lcudart

commonBMT.o : commonBMT.cpp
	$(CHARMC) $(CHARM_INC) $(CPP_FLAGS) $< -o $@

himenocuda.o : himenocuda.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $(CHARM_INC) -o himenocuda.o himenocuda.cu

himeno.decl.h : himeno.ci
	$(CHARMC) $(MVCC_INC) himeno.ci

clean:
	rm -f *.decl.h *.def.h *.o himeno charmrun
himeno.o : himeno.C himeno.decl.h himeno.h
	$(CHARMC) $(NVCC_INC) -O2 -c himeno.C



