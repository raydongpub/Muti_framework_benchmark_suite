IVM_MOD_TARGET   = nb_ivm
IVM_MOD_SRC      = nnivm.cpp

IVM_MOD_FTARGET  = $(addprefix $(ROOT_DIR)/, $(IVM_MOD_TARGET))
IVM_MOD_PTARGET  = $(addprefix $(IVM_DIR)/, $(IVM_MOD_TARGET))
IVM_MOD_PSRC     = $(addprefix $(IVM_DIR)/, $(IVM_MOD_SRC))
IVM_MOD_POBJ     = $(addprefix $(IVM_DIR)/obj/, $(IVM_MOD_SRC:.cpp=.o))
IVM_MOD_POBJ    += $(patsubst $(ROOT_DIR)/%, $(IVM_DIR)/obj/%, $(MAIN_IVM_SRC:.cpp=.o))
IVM_MOD_CINC     = -D_IVM_MOD -I$(ROOT_DIR) -I$(IVM_DIR) -I$(IVM_INC_DIR) -I$(MPI_DIR)
IVM_MOD_CLEAN    = IVM_MOD_CLEAN

