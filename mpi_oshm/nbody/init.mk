
##### Directories #####
CUDA_LDIR        = /usr/local/cuda
OSHM_LDIR        = /usr/local/openshmem#/cluster/software/openshmem/openshmem-1.2
MPI_LDIR         = /usr/local/openmpi#/cluster/rcss-spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/openmpi-2.0.1-mtpjjw5dmjopgjj7rucvvjwrnzyw7sls
GSN_LDIR         = /usr/local/gasnet-1.28.2#/cluster/rcss-spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/gasnet-1.24.0-nokdtfuj5uwsgqqn5efvyf2gycm7lp3k
##### IVM-PART #####
IVM_ROOT_DIR    = /home/ksajjapongse/DGAS
IVM_IPCVM_DIR   = $(IVM_ROOT_DIR)/ipcvm
IVM_IPCVM_BIN_DIR = $(IVM_IPCVM_DIR)/bin
IVM_LM_DIR      = $(IVM_IPCVM_DIR)/lm
IVM_CH_DIR      = $(IVM_IPCVM_DIR)/ch
IVM_LDIR        = $(IVM_IPCVM_DIR)/ivm

##### Dependencies directory #####
IBV_INC_DIR     = /usr/include/infiniband/
IBV_LIB_DIR     = /usr/lib64/
RDMACM_INC_DIR  = /usr/include/rdma
RDMACM_LIB_DIR  = /usr/lib64

##### Options #####
OPT_DISABLE_LM_RDMA = 
OPT_ENABLE_TRACE    = 1

ROOT_DIR        = $(shell pwd)
MODULES_DIR     = $(ROOT_DIR)/modules
MPI_DIR         = $(MODULES_DIR)/mpi
MPICUDA_DIR     = $(MODULES_DIR)/mpi_cuda
OSHM_DIR        = $(MODULES_DIR)/oshm
OSHMCUDA_DIR    = $(MODULES_DIR)/oshm_cuda
ST_DIR          = $(MODULES_DIR)/st
IVM_DIR         = $(MODULES_DIR)/ivm
IVM_PC_DIR      = $(MODULES_DIR)/ivm_pc
IVM_CUDA_DIR    = $(MODULES_DIR)/ivm_cuda
CUDA_INC_DIR    = $(CUDA_LDIR)/include
CUDA_LIB_DIR    = $(CUDA_LDIR)/lib64
#MPI_INC_DIR     = $(MPI_LDIR)/include
MPI_INC_DIR     = $(MPI_LDIR)/include
MPI_LIB_DIR     = $(MPI_LDIR)/lib
OSHM_INC_DIR    = $(OSHM_LDIR)/include
OSHM_LIB_DIR    = $(OSHM_LDIR)/lib
IVM_INC_DIR     = $(IVM_LDIR)
IVM_LIB_DIR     = $(IVM_LDIR)/bin
IVM_LIB         = $(IVM_LIB_DIR)/libivm.so
GSN_INC_DIR     = $(GSN_LDIR)/include
GSN_LIB_DIR     = $(GSN_LDIR)/lib


##### Compiler #####
CPP             = g++
NVCC            = nvcc
MPI_CPP         = mpicxx
OSHM_CPP        = oshc++
#C_COMMON_FLAGS  = -c -O2 -D_USE_STATIC_LIB -D_DOUBLE_PRECISION -I$(MPI_INC_DIR)
C_COMMON_FLAGS  = -c -O3 -D_USE_STATIC_LIB -I$(MPI_INC_DIR)
C_WARNING_FLAGS = -Wall
C_SHARED_FLAGS  = -fPIC
C_DEP_FLAGS     = -MD
C_DEBUG_FLAGS   = -g
#C_IVM_FLAGS     = -c -O2 -D_USE_STATIC_LIB -D_DOUBLE_PRECISION -I$(MPI_INC_DIR) 
C_IVM_FLAGS     = -c -O3 -D_USE_STATIC_LIB -I$(MPI_INC_DIR) 

##### Linker #####
CPP_LD          = $(CPP)
NVCC_LD         = $(NVCC)
MPI_LD          = $(MPI_CPP)
OSHM_LD         = $(OSHM_CPP)
LD_PFLAGS       = 
LD_SFLAGS       = 
#LD_IVM_PFLAGS   = -arch=sm_20 --cudart=shared -L$(CUDA_LIB_DIR) -L$(IVM_LIB_DIR) -L$(IBV_LIB_DIR) -L$(RDMACM_LIB_DIR) -I$(MPI_LIB_DIR)
LD_IVM_PFLAGS   = --cudart=shared -L$(CUDA_LIB_DIR) -L$(IVM_LIB_DIR) -L$(IBV_LIB_DIR) -L$(RDMACM_LIB_DIR) -I$(MPI_LIB_DIR)
#LD_IVM_SFLAGS   = --linker-options="-Wl,-Bstatic -livm -Wl,-Bdynamic -lrt -lpthread -libverbs -lrdmacm"
LD_IVM_SFLAGS   = --linker-options="-L$(CUDA_LIB_DIR) -Bstatic -livm -Bdynamic -lrt -lpthread"
#LD_IVM_CUDA_SFLAGS = -livm -lrt -lpthread -libverbs -lrdmacm
LD_IVM_CUDA_SFLAGS = -livm -lrt -lpthread
LD_SHARED_FLAGS = --shared

##### Utils #####
RM              = rm
RM_FLAGS        = -f
CP              = cp
CP_FLAGS        = 
AR              = ar
AR_FLAGS        = rcs
