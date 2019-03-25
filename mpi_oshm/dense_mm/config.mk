
MATRIXDATA_TARGET  = matrixset
SHOWPAR_TARGET     = show

MATRIXDATA_SRC     = Matrixcreate.cpp \
                     matdataset.cpp

SHOWPAR_SRC        = ShowPar.cpp      \
                     matdataset.cpp

#MAIN_TARGET        = main
#MAIN_SRC           = main.cpp matdataset.cpp
#MAIN_PTARGET       = $(addprefix $(ROOT_DIR)/, $(MAIN_TARGET))
#MAIN_POBJ          = $(addprefix $(ROOT_DIR)/obj/, $(MAIN_SRC:.cpp=.o))

MATRIXDATA_PTARGET = $(addprefix $(ROOT_DIR)/, $(MATRIXDATA_TARGET))
MATRIXDATA_POBJ    = $(addprefix $(ROOT_DIR)/obj/, $(MATRIXDATA_SRC:.cpp=.o))
SHOWPAR_PTARGET    = $(addprefix $(ROOT_DIR)/, $(SHOWPAR_TARGET))
SHOWPAR_POBJ       = $(addprefix $(ROOT_DIR)/obj/, $(SHOWPAR_SRC:.cpp=.o))

COMMON_CLEAN       = common_clean 
