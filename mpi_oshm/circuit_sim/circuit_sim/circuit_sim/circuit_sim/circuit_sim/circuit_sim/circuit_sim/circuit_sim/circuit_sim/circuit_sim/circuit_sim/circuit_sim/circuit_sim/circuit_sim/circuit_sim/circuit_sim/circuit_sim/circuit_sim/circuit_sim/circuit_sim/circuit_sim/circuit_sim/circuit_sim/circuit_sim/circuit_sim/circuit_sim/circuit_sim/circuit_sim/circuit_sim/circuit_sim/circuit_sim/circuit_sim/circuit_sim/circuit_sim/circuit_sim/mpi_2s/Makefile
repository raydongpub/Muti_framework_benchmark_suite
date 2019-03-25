TARGET       = circuitcuda
SRC          = main.cu
OBJ          = $(SRC:.cu=.o)

ROOT_DIR     = $(shell pwd)
#MPI_DIR      = /cluster/rcss-spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/openmpi-2.0.1-q2o7rcbmowk72uov5asfqxfsixcde4he
#CUDA_DIR     = /usr/local/cuda-7.5
CUDA_DIR     = /usr/local/cuda
MPI_DIR      = /usr/local/openmpi
CUDA_LIB_DIR = $(CUDA_DIR)/lib64
CUDA_INC_DIR = $(CUDA_DIR)/include
MPI_LIB_DIR  = $(MPI_DIR)/lib
MPI_INC_DIR  = $(MPI_DIR)/include

NVCC         = nvcc
NVCC_FLAGS   = -c -g -O2 -arch=sm_20
NVCC_LDFLAGS = -arch=sm_20 --cudart=shared
C_COMMON_FLAGS   = -c -O2 -I$(MPI_INC_DIR) -I$(MPI_LIB_DIR) #-D_DEBUG 
C_WARNNING_FLAGS = -Wall
C_SHARED_FLAGS   = -fPIC
C_DEP_FLAGS      = -MD
C_DEBUG_FLAGS    = -g
MPICUDA_MOD_NVCCFLAGS= -arch=sm_20 --cudart=shared
MPICUDA_MOD_LDPFLAGS = -L$(MPI_LIB_DIR) -L$(MPI_INC_DIR)
MPICUDA_MOD_LDSFLAGS = -lmpi -lmpi_cxx #-lmpl

MPICXX       = mpicxx
MPICXX_FLAGS = -O2 -g
M            = rm
RM_FLAGS     = -f
CP           = cp
CP_FLAGS     =


MPICUDA_MOD_FTARGET = $(addprfix )

all: $(TARGET)

$(TARGET) : $(OBJ)
	$(NVCC) $(MPICUDACUDA_MOD_NVCCFLAGS) $(MPICUDA_MOD_LDPFLAGS) $^ -o $@ $(MPICUDA_MOD_LDSFLAGS) 
	@printf "Built successfully.\n\n"
$(OBJ): $(SRC)
	$(NVCC) $(C_COMMON_FLAGS) $(C_DEBUG_FLAGS) $(NVCC_LDFLAGS) $(SRC)
	@printf "Compile successfully.\n\n"

clean:
	$(RM) -f $(TARGET)
	$(RM) -f $(OBJ)
	@printf "Cleaned successfully.\n\n"

