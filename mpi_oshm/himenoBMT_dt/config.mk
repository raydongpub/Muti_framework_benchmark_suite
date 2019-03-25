TMAT_TARGET  = tmat

TMAT_SRC     = commonBMT.cpp \
               tmat.cpp

TMAT_PTARGET = $(addprefix $(ROOT_DIR)/, $(TMAT_TARGET))
TMAT_POBJ    = $(addprefix $(ROOT_DIR)/obj/, $(TMAT_SRC:.cpp=.o))

COMMON_CLEAN = common_clean

