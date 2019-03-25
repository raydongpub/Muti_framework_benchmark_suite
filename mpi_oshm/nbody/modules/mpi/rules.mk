
.PHONY: $(MPI_MOD_CLEAN)

$(MPI_MOD_PTARGET) : $(MPI_MOD_POBJ)
	$(MPI_LD) $(LD_PFLAGS) $^ -o $@ $(LD_SFLAGS)
	@printf "\n***** MPI modules built successfully *****\n\n"

$(MPI_DIR)/obj/%.o : $(MPI_DIR)/%.cpp
	$(MPI_CPP) $(C_COMMON_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $(MPI_MOD_CINC)  $< -o $@

$(MPI_DIR)/obj/%.o : $(ROOT_DIR)/%.cpp
	$(CPP) $(C_COMMON_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $(MPI_MOD_CINC) $< -o $@

include $(wildcard $(MPI_DIR)/obj/*.d)

$(MPI_MOD_CLEAN) :
	$(RM) $(RM_FLAGS) $(MPI_MOD_POBJ) 
	$(RM) $(RM_FLAGS) $(MPI_MOD_PTARGET)
	$(RM) $(RM_FLAGS) $(MPI_DIR)/obj/*.d

