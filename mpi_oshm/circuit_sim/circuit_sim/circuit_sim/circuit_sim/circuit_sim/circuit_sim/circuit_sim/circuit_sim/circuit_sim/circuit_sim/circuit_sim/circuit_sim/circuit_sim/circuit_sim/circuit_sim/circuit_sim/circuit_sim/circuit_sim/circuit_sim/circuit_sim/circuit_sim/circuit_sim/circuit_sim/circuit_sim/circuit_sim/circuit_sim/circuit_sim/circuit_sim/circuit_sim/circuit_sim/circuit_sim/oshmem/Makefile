TARGET       = circuitcuda
SRC          = main.cu
OBJ          = $(SRC:.cu=.o)

ROOT_DIR     = $(shell pwd)
CUDA_DIR     = /usr/local/cuda
MPI_DIR      = /usr/local/openmpi
OSHM_DIR      = /usr/local/osh
CUDA_LIB_DIR = $(CUDA_DIR)/lib64
CUDA_INC_DIR = $(CUDA_DIR)/include
MPI_LIB_DIR  = $(MPI_DIR)/lib
MPI_INC_DIR  = $(MPI_DIR)/include
OSHM_LIB_DIR  = $(OSHM_DIR)/lib
OSHM_INC_DIR  = $(OSHM_DIR)/include

NVCC         = nvcc
NVCC_FLAGS   = -c -g -O2 -arch=sm_20
NVCC_LDFLAGS = -arch=sm_20 --cudart=shared
OSHM_CPP     = oshCC#oshCC
C_COMMON_FLAGS   = -c -O2 -D_DEBUG -I$(MPI_INC_DIR) -I$(MPI_LIB_DIR)
C_WARNNING_FLAGS = -Wall
C_SHARED_FLAGS   = -fPIC
C_DEP_FLAGS      = -MD
C_DEBUG_FLAGS    = -g
CUDA_MOD_NVCCFLAGS= -arch=sm_20 --cudart=shared
MPICUDA_MOD_LDPFLAGS = -L$(MPI_LIB_DIR) -L$(MPI_INC_DIR)
MPICUDA_MOD_LDSFLAGS = -lmpi -lmpi_cxx -lmpl
OSHMCUDA_MOD_LDPFLAGS = -L$(OSHM_LIB_DIR) -L$(OSHM_INC_DIR)
CUDA_MOD_LDPFLAGS = -L$(CUDA_LIB_DIR) -L$(CUDA_INC_DIR)

MPICXX       = mpicxx
MPICXX_FLAGS = -O2 -g
M            = rm
RM_FLAGS     = -f
CP           = cp
CP_FLAGS     =


MPICUDA_MOD_FTARGET = $(addprfix )

all: $(TARGET)

$(TARGET) : $(OBJ)
	$(OSHM_CPP) $(OSHMCUDA_MOD_LDPFLAGS) $(CUDA_MOD_LDPFLAGS) $^ -o $@ -lcudart
	@printf "Built successfully.\n\n"
#	$(OSHM_CPP) $(CUDA_MOD_NVCCFLAGS) $(OSHMCUDA_MOD_LDPFLAGS) $^ -o $@  
$(OBJ): $(SRC)
	$(NVCC) $(C_COMMON_FLAGS) $(C_DEBUG_FLAGS) $(NVCC_LDFLAGS) $(OSHMCUDA_MOD_LDPFLAGS) $(SRC)
	@printf "Compile successfully.\n\n"

clean:
	$(RM) -f $(TARGET)
	$(RM) -f $(OBJ)
	@printf "Cleaned successfully.\n\n"

