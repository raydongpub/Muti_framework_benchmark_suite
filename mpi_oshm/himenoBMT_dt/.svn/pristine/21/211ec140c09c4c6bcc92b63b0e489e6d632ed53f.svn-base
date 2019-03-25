OSHM_MOD_TARGET   = bmt_oshm
OSHM_MOD_SRC      = oshmBMT.cpp

OSHM_MOD_FTARGET  = $(addprefix $(ROOT_DIR)/, $(OSHM_MOD_TARGET))
OSHM_MOD_PTARGET  = $(addprefix $(OSHM_DIR)/, $(OSHM_MOD_TARGET))
OSHM_MOD_PSRC     = $(addprefix $(OSHM_DIR)/, $(OSHM_MOD_SRC))
OSHM_MOD_POBJ     = $(addprefix $(OSHM_DIR)/obj/, $(OSHM_MOD_SRC:.cpp=.o))
OSHM_MOD_POBJ    += $(patsubst $(ROOT_DIR)/%, $(OSHM_DIR)/obj/%, $(MAIN_SRC:.cpp=.o))
OSHM_MOD_CINC     = -D_OSHM_MOD -I$(ROOT_DIR) -I$(OSHM_DIR)
OSHM_MOD_CLEAN    = OSHM_MOD_CLEAN

