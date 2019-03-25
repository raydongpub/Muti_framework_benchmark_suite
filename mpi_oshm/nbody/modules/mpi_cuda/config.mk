MPICUDA_MOD_TARGET   = nb_mpicuda
MPICUDA_MOD_SRC      = nnMpiCuda.cpp      \
                       nnMpiCudaKernel.cu

MPICUDA_MOD_FTARGET  = $(addprefix $(ROOT_DIR)/, $(MPICUDA_MOD_TARGET))
MPICUDA_MOD_PTARGET  = $(addprefix $(MPICUDA_DIR)/, $(MPICUDA_MOD_TARGET))
MPICUDA_MOD_PSRC     = $(addprefix $(MPICUDA_DIR)/, $(MPICUDA_MOD_SRC))
MPICUDA_MOD_SRC_1    = $(MPICUDA_MOD_SRC:.cpp=.o)
MPICUDA_MOD_SRC_2    = $(MPICUDA_MOD_SRC_1:.cu=.o)
MPICUDA_MOD_POBJ     = $(addprefix $(MPICUDA_DIR)/obj/, $(MPICUDA_MOD_SRC_2))
MPICUDA_MOD_POBJ    += $(patsubst $(ROOT_DIR)/%, $(MPICUDA_DIR)/obj/%, $(MAIN_SRC:.cpp=.o))
MPICUDA_MOD_CINC     = -I$(ROOT_DIR) -I$(MPICUDA_DIR) -I$(CUDA_INC_DIR)
MPICUDA_MOD_NVCCFLAGS= -arch=sm_30 --cudart=shared 
MPICUDA_MOD_LDPFLAGS = -L$(MPI_LIB_DIR) 
#MPICUDA_MOD_LDSFLAGS = -lmpich -lmpl
MPICUDA_MOD_LDSFLAGS = -lmpi_cxx -lmpi

MPICUDA_MOD_CLEAN    = MPICUDA_MOD_CLEAN

