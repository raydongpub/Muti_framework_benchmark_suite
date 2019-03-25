IVM_CUDA_MOD_TARGET   = nb_ivmcuda
IVM_CUDA_MOD_SRC      = nnivmcuda.cpp    \
                        nnIVMCudaKernel.cu

IVM_CUDA_MOD_FTARGET  = $(addprefix $(ROOT_DIR)/, $(IVM_CUDA_MOD_TARGET))
IVM_CUDA_MOD_PTARGET  = $(addprefix $(IVM_CUDA_DIR)/, $(IVM_CUDA_MOD_TARGET))
IVM_CUDA_MOD_PSRC     = $(addprefix $(IVM_CUDA_DIR)/, $(IVM_CUDA_MOD_SRC))
IVM_CUDA_MOD_SRC_1    = $(IVM_CUDA_MOD_SRC:.cpp=.o)
IVM_CUDA_MOD_SRC_2    = $(IVM_CUDA_MOD_SRC_1:.cu=.o)
IVM_CUDA_MOD_POBJ     = $(addprefix $(IVM_CUDA_DIR)/obj/, $(IVM_CUDA_MOD_SRC_2:.cpp=.o))
IVM_CUDA_MOD_POBJ    += $(patsubst $(ROOT_DIR)/%, $(IVM_CUDA_DIR)/obj/%, $(MAIN_IVM_SRC:.cpp=.o))
IVM_CUDA_MOD_CINC     = -D_IVM_MOD -I$(ROOT_DIR) -I$(IVM_CUDA_DIR) -I$(IVM_INC_DIR) -I$(CUDA_INC_DIR)
IVM_CUDA_MOD_NVCCFLAGS= -arch=sm_20 --cudart=shared
IVM_CUDA_MOD_LDPFLAGS = $(IVM_CUDA_MOD_NVCCFLAGS)
IVM_CUDA_MOD_CLEAN    = IVM_CUDA_MOD_CLEAN

