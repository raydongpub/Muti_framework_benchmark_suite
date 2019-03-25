##### Commands #####
CHARMC = charmc
NVCC   = nvcc
CPP    = g++
##### Compiler #####

##### LoadBalance setting #####
CHARM_LB=-module CommonLBs
#CHARM_LB=-module CommonLBs

##### Object #####
OBJS =Config.o NbodyConfig.o ParticleDataset.o nbodycuda.o nbody.o 
##### Directry #####
NVCC_DIR =/usr/local/cuda

##### LIBINC #####
NVCC_LIB  = -L$(NVCC_DIR)/lib64
NVCC_INC  = -I$(NVCC_DIR)/include
CHARM_INC = -I/home/rgu/software/charm/mpi-linux-x86_64/include#-I/cluster/software/charm/charm-6.7.0-net/include#-I/home/ruidong/Documents/software/charm/net-linux-x86_64/include

##### Flags #####
NVCC_FLAGS = -O2 -c #-use_fast_math #-device-debug -deviceemu
#CPP_FLAGS  = -cpp-option -c -O2 -D_USE_STATIC_LIB -Wall -fPIC -g
CPP_FLAGS  = -cpp-option -c -O2 -D_USE_STATIC_LIB -Wall -fPIC -g #-D_DOUBLE_PRECISION

all : nbody

nbody : $(OBJS)
	$(CHARMC) $(CHARM_LB) -language charm++ -o nbody $(OBJS) $(NVCC_LIB) -lcuda -lcudart

Config.o : Config.cpp
	$(CHARMC) $(CPP_FLAGS) $< -o $@


NbodyConfig.o : NbodyConfig.cpp
	$(CHARMC) $(CPP_FLAGS) $< -o $@


ParticleDataset.o : ParticleDataset.cpp
	$(CHARMC) $(CHARM_INC) $(CPP_FLAGS) $< -o $@

nbodycuda.o : nbodycuda.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $(CHARM_INC) -o nbodycuda.o nbodycuda.cu
#	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) $(CHARM_INC) -o nbodycuda.o nbodycuda.cu

nbody.decl.h : nbody.ci
	$(CHARMC) $(MVCC_INC) nbody.ci

clean:
	rm -f *.decl.h *.def.h *.o nbody charmrun
nbody.o : nbody.C nbody.decl.h 
	$(CHARMC) $(NVCC_INC) -O2 -c nbody.C  
#	$(CHARMC) $(NVCC_INC) -D_COMM -O2 -c nbody.C  




