OSHMCUDA_MOD_TARGET   = mm_oshmcuda
OSHMCUDA_MOD_SRC      = mmoshmcuda.cpp  \
                        mmoshmcudakernel.cu

OSHMCUDA_MOD_FTARGET  = $(addprefix $(ROOT_DIR)/, $(OSHMCUDA_MOD_TARGET))
OSHMCUDA_MOD_PTARGET  = $(addprefix $(OSHMCUDA_DIR)/, $(OSHMCUDA_MOD_TARGET))
OSHMCUDA_MOD_PSRC     = $(addprefix $(OSHMCUDA_DIR)/, $(OSHMCUDA_MOD_SRC))
OSHMCUDA_MOD_SRC_1    = $(OSHMCUDA_MOD_SRC:.cpp=.o)
OSHMCUDA_MOD_SRC_2    = $(OSHMCUDA_MOD_SRC_1:.cu=.o)
OSHMCUDA_MOD_POBJ     = $(addprefix $(OSHMCUDA_DIR)/obj/,$(OSHMCUDA_MOD_SRC_2))
OSHMCUDA_MOD_POBJ    += $(patsubst $(ROOT_DIR)/%, $(OSHMCUDA_DIR)/obj/%, $(MAIN_SRC:.cpp=.o))
OSHMCUDA_MOD_CINC     = -D_OSHM_MOD -I$(ROOT_DIR) -I$(OSHM_DIR) -I$(CUDA_INC_DIR)
OSHMCUDA_MOD_NVCCFLAGS= -arch=sm_20 --cudart=shared
OSHMCUDA_MOD_LDPFLAGS = -L$(OSHM_LIB_DIR) -L$(MPI_LIB_DIR) -L$(GSN_LIB_DIR) -L$(CUDA_LIB_DIR)
OSHMCUDA_MOD_LDSFLAGS = -lcudart -lmpi_cxx -lmpi#-Xlinker="-Bdynamic -lopenshmem -Bstatic -lgasnet-mpi-par -lammpi -lmpi_cxx -lmpi"
#OSHMCUDA_MOD_LDSFLAGS = -I$(OSHM_LIB_DIR) -lopenshmem -lgasnet-mpi-par -lammpi -I$(MPI_LIB_DIR) -lmpich -lmpl
OSHMCUDA_MOD_CLEAN    = OSHMCUDA_MOD_CLEAN

