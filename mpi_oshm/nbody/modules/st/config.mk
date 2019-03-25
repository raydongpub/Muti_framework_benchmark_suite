ST_MOD_TARGET   = nb_st
ST_MOD_SRC      = nnst.cpp

ST_MOD_FTARGET  = $(addprefix $(ROOT_DIR)/, $(ST_MOD_TARGET))
ST_MOD_PTARGET  = $(addprefix $(ST_DIR)/, $(ST_MOD_TARGET))
ST_MOD_PSRC     = $(addprefix $(ST_DIR)/, $(ST_MOD_SRC))
ST_MOD_POBJ     = $(addprefix $(ST_DIR)/obj/, $(ST_MOD_SRC:.cpp=.o))
ST_MOD_POBJ    += $(patsubst $(ROOT_DIR)/%, $(ST_DIR)/obj/%, $(MAIN_SRC:.cpp=.o))
ST_MOD_CINC     = -I$(ROOT_DIR) -I$(ST_DIR)
ST_MOD_LDPFLAGS =
ST_MOD_LDSFLAGS = -lpthread

ST_MOD_CLEAN    = ST_MOD_CLEAN

