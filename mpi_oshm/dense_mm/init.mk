#### Directories ####
CUDA_LDIR        = /usr/local/cuda
OSHM_LDIR        = /usr/local/openshmem#/cluster/software/openshmem/openshmem-1.2
MPI_LDIR         = /usr/local/openmpi#/cluster/rcss-spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/openmpi-2.0.1-mtpjjw5dmjopgjj7rucvvjwrnzyw7sls
GSN_LDIR         = /usr/local/gasnet-1.28.2#/cluster/rcss-spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/gasnet-1.24.0-nokdtfuj5uwsgqqn5efvyf2gycm7lp3k

ROOT_DIR         = $(shell pwd)
MODULES_DIR      = $(ROOT_DIR)/modules
ST_DIR           = $(MODULES_DIR)/st
OSHM_DIR         = $(MODULES_DIR)/oshm
OSHMCUDA_DIR     = $(MODULES_DIR)/oshm_cuda
MPICUDA_DIR      = $(MODULES_DIR)/mpi_cuda
OSHM_INC_DIR     = $(OSHM_LDIR)/include
OSHM_LIB_DIR     = $(OSHM_LDIR)/lib
MPI_INC_DIR      = $(MPI_LDIR)/include
MPI_LIB_DIR      = $(MPI_LDIR)/lib
GSN_INC_DIR      = $(GSN_LDIR)/include
GSN_LIB_DIR      = $(GSN_LDIR)/lib
CUDA_INC_DIR     = $(CUDA_LDIR)/include
CUDA_LIB_DIR     = $(CUDA_LDIR)/lib64

#### Compiler ####
CPP              = g++
NVCC             = $(CUDA_LDIR)/bin/nvcc
MPI_CPP          = mpicc
#C_COMMON_FLAGS   = -c -O2 -D_USE_STATIC_LIB -D_DOUBLE_PRECISION -I$(MPI_INC_DIR)
C_COMMON_FLAGS   = -c -O2 -D_USE_STATIC_LIB -I$(MPI_INC_DIR)
C_WARNNING_FLAGS = -Wall
C_SHARED_FLAGS   = -fPIC
C_DEP_FLAGS      = -MD
C_DEBUG_FLAGS    = -g
OSHM_CPP         = oshc++

#### Linker ####
CPP_LD           = $(CPP)
NVCC_LD          = $(NVCC)
MPI_LD           = $(MPI_CPP)
LD_PFLAGS        = 
LD_SFLAGS        =
OSHM_LD          = $(OSHM_CPP)

#### Utils ####
RM               = rm 
RM_FLAGS         = -f
CP               = cp
CP_FLAGS         = 

