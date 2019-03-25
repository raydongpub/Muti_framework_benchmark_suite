IVM_PC_MOD_TARGET   = nb_ivmpc
IVM_PC_MOD_SRC      = nnivmpc.cpp

IVM_PC_MOD_FTARGET  = $(addprefix $(ROOT_DIR)/, $(IVM_PC_MOD_TARGET))
IVM_PC_MOD_PTARGET  = $(addprefix $(IVM_PC_DIR)/, $(IVM_PC_MOD_TARGET))
IVM_PC_MOD_PSRC     = $(addprefix $(IVM_PC_DIR)/, $(IVM_PC_MOD_SRC))
IVM_PC_MOD_POBJ     = $(addprefix $(IVM_PC_DIR)/obj/, $(IVM_PC_MOD_SRC:.cpp=.o))
IVM_PC_MOD_POBJ    += $(patsubst $(ROOT_DIR)/%, $(IVM_PC_DIR)/obj/%, $(MAIN_IVM_SRC:.cpp=.o))
IVM_PC_MOD_CINC     = -D_IVM_MOD -I$(ROOT_DIR) -I$(IVM_PC_DIR) -I$(IVM_INC_DIR) -I$(MPI_DIR)
IVM_PC_MOD_CLEAN    = IVM_PC_MOD_CLEAN

