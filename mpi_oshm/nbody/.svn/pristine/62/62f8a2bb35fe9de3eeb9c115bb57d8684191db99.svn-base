
.PHONY: $(ST_MOD_CLEAN)

$(ST_MOD_PTARGET) : $(ST_MOD_POBJ)
	$(MPI_LD) $(LD_PFLAGS) $(ST_MOD_LDPFLAGS) $^ -o $@ $(LD_SFLAGS) $(ST_MOD_LDSFLAGS)
	@printf "\n***** Serial modules built successfully *****\n\n"

$(ST_DIR)/obj/%.o : $(ST_DIR)/%.cpp
	$(CPP) $(C_COMMON_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $(ST_MOD_CINC)  $< -o $@

$(ST_DIR)/obj/%.o : $(ROOT_DIR)/%.cpp
	$(CPP) $(C_COMMON_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $(ST_MOD_CINC) $< -o $@

include $(wildcard $(ST_DIR)/obj/*.d)

$(ST_MOD_CLEAN) :
	$(RM) $(RM_FLAGS) $(ST_MOD_POBJ) 
	$(RM) $(RM_FLAGS) $(ST_MOD_PTARGET)
	$(RM) $(RM_FLAGS) $(ST_DIR)/obj/*.d

