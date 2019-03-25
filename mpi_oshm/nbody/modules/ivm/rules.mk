
.PHONY: $(IVM_MOD_CLEAN)

$(IVM_MOD_PTARGET) : $(IVM_MOD_POBJ)
	$(NVCC_LD) $(LD_IVM_PFLAGS) $^ -o $@ $(LD_IVM_SFLAGS)
	@printf "\n***** IVM modules built successfully *****\n\n"

$(IVM_DIR)/obj/%.o : $(IVM_DIR)/%.cpp
	$(CPP) $(C_IVM_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $(IVM_MOD_CINC)  $< -o $@

$(IVM_DIR)/obj/%.o : $(ROOT_DIR)/%.cpp
	$(CPP) $(C_IVM_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $(IVM_MOD_CINC) $< -o $@

include $(wildcard $(IVM_DIR)/obj/*.d)

$(IVM_MOD_CLEAN) :
	$(RM) $(RM_FLAGS) $(IVM_MOD_POBJ) 
	$(RM) $(RM_FLAGS) $(IVM_MOD_PTARGET)
	$(RM) $(RM_FLAGS) $(IVM_DIR)/obj/*.d

