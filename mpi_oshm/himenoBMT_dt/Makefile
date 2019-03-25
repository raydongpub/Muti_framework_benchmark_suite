include init.mk
include common.mk
include $(MPI_DIR)/config.mk
include $(MPICUDA_DIR)/config.mk
include $(OSHM_DIR)/config.mk
include $(OSHMCUDA_DIR)/config.mk
include $(ROOT_DIR)/config.mk

.PHONY: all mpi mpi_cuda oshm_cuda st clean show

#all: mpi mpi_cuda oshm oshm_cuda common
all: mpi mpi_cuda oshm_cuda common
	@printf "Build successfully\n\n"

clean: $(MPI_MOD_CLEAN) $(MPICUDA_MOD_CLEAN) $(OSHM_MOD_CLEAN) $(OSHMCUDA_MOD_CLEAN) $(ST_MOD_CLEAN) $(COMMON_CLEAN)
	$(RM) $(RM_FLAGS) $(MPI_MOD_FTARGET)
	$(RM) $(RM_FLAGS) $(MPICUDA_MOD_FTARGET)
	$(RM) $(RM_FLAGS) $(OSHM_MOD_FTARGET)
	$(RM) $(RM_FLAGS) $(OSHMCUDA_MOD_FTARGET)
	@printf "Clean complete\n\n"

show:
	@printf "$(DATASET_POBJ)\n\n$(SHOWPAR_POBJ)\n\n$(DATASET_PTARGET)\n\n$(SHOWPAR_PTARGET)\n\n"

mpi: $(MPI_MOD_FTARGET)
	@printf "\n"

mpi_cuda: $(MPICUDA_MOD_FTARGET)
	@printf "\n"

oshm: $(OSHM_MOD_FTARGET)
	@printf "\n"

oshm_cuda: $(OSHMCUDA_MOD_FTARGET)
	@printf "\n"

st: $(ST_MOD_FTARGET)
	@printf "\n"

common: $(TMAT_PTARGET)
	@printf "\n"

$(MPI_MOD_FTARGET) : $(MPI_MOD_PTARGET)
	$(CP) $(CP_FLAGS) $< $@

$(MPICUDA_MOD_FTARGET) : $(MPICUDA_MOD_PTARGET)
	$(CP) $(CP_FLAGS) $< $@

$(OSHM_MOD_FTARGET) : $(OSHM_MOD_PTARGET)
	$(CP) $(CP_FLAGS) $< $@

$(OSHMCUDA_MOD_FTARGET) : $(OSHMCUDA_MOD_PTARGET)
	$(CP) $(CP_FLAGS) $< $@

$(ST_MOD_FTARGET) : $(ST_MOD_PTARGET)
	$(CP) $(CP_FLAGS) $< $@

include $(MPI_DIR)/rules.mk
include $(MPICUDA_DIR)/rules.mk
include $(OSHM_DIR)/rules.mk
include $(OSHMCUDA_DIR)/rules.mk
include $(ROOT_DIR)/rules.mk
