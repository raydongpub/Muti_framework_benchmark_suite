MPI_MOD_TARGET   = bmt_mpi
MPI_MOD_SRC      = mpiBMT.cpp

MPI_MOD_FTARGET  = $(addprefix $(ROOT_DIR)/, $(MPI_MOD_TARGET))
MPI_MOD_PTARGET  = $(addprefix $(MPI_DIR)/, $(MPI_MOD_TARGET))
MPI_MOD_PSRC     = $(addprefix $(MPI_DIR)/, $(MPI_MOD_SRC))
MPI_MOD_POBJ     = $(addprefix $(MPI_DIR)/obj/, $(MPI_MOD_SRC:.cpp=.o))
MPI_MOD_POBJ    += $(patsubst $(ROOT_DIR)/%, $(MPI_DIR)/obj/%, $(MAIN_SRC:.cpp=.o))
MPI_MOD_CINC     = -I$(ROOT_DIR) -I$(MPI_DIR)
MPI_MOD_CLEAN    = MPI_MOD_CLEAN

