TARGET       = nbodycuda
SRC          = main.cu             \
               nnMpiCudaKernel.cu  \
               Config.cpp          \
               NbodyConfig.cpp     \
               ParticleDataset.cpp 
OBJ_1        = $(SRC:.cu=.o)
OBJ_2        = $(OBJ_1:.cpp=.o)

ROOT_DIR     = $(shell pwd)
CUDA_DIR     = /usr/local/cuda
MPI_DIR      = /usr/local/openmpi
CUDA_LIB_DIR = $(CUDA_DIR)/lib64
CUDA_INC_DIR = $(CUDA_DIR)/include
MPI_LIB_DIR  = $(MPI_DIR)/lib
MPI_INC_DIR  = $(MPI_DIR)/include

NVCC         = nvcc
NVCC_FLAGS   = -c -g -O2 -arch=sm_30
NVCC_LDFLAGS = -arch=sm_30 --cudart=shared
CPP              = g++
C_COMMON_FLAGS   = -c -O2 -D_USE_STATIC_LIB -I$(MPI_INC_DIR) -I$(MPI_LIB_DIR)#-D_DOUBLE_PRECISION 
C_WARNNING_FLAGS = -Wall
C_SHARED_FLAGS   = -fPIC
C_DEP_FLAGS      = -MD
C_DEBUG_FLAGS    = -g
MPICUDA_MOD_NVCCFLAGS= -arch=sm_30 --cudart=shared
MPICUDA_MOD_LDPFLAGS = -L$(MPI_LIB_DIR) -L$(MPI_INC_DIR)
MPICUDA_MOD_LDSFLAGS = -lmpi -lmpi_cxx

MPICXX       = mpicxx
MPICXX_FLAGS = -O2 -g
M            = rm
RM_FLAGS     = -f
CP           = cp
CP_FLAGS     =


MPICUDA_MOD_FTARGET = $(addprfix )

all: $(TARGET)

$(TARGET) : $(OBJ_2)
	$(NVCC) $(MPICUDACUDA_MOD_NVCCFLAGS) $(MPICUDA_MOD_LDPFLAGS) $^ -o $@ $(MPICUDA_MOD_LDSFLAGS) 
	@printf "Built successfully.\n\n"
$(OBJ_2): $(SRC)
	$(NVCC) $(C_COMMON_FLAGS) $(C_DEBUG_FLAGS) $(NVCC_LDFLAGS) $(SRC)
	@printf "Compile successfully.\n\n"

clean:
	$(RM) -f $(TARGET)
	$(RM) -f $(OBJ_2)
	@printf "Cleaned successfully.\n\n"

