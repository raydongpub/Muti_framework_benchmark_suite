DATASET_TARGET  = dataset
SHOWPAR_TARGET  = show

DATASET_SRC     = Dataset.cpp         \
                  ParticleDataset.cpp
SHOWPAR_SRC     = ShowPar.cpp         \
                  ParticleDataset.cpp

DATASET_PTARGET = $(addprefix $(ROOT_DIR)/, $(DATASET_TARGET))
SHOWPAR_PTARGET = $(addprefix $(ROOT_DIR)/, $(SHOWPAR_TARGET))
DATASET_POBJ    = $(addprefix $(ROOT_DIR)/obj/, $(DATASET_SRC:.cpp=.o))
SHOWPAR_POBJ    = $(addprefix $(ROOT_DIR)/obj/, $(SHOWPAR_SRC:.cpp=.o))

COMMON_CLEAN    = common_clean

