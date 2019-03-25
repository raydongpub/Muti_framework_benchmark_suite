.PHONY: OSHM_MOD_CLEAN

$(OSHM_MOD_PTARGET): $(OSHM_MOD_POBJ)
	$(OSHM_LD) $(LD_PFLAGS) $^ -o $@ $(LD_SFLAGS)
	@printf "\n***** OSHM modules built successfully *****\n\n"

$(OSHM_DIR)/obj/%.o: $(OSHM_DIR)/%.cpp
	$(OSHM_CPP) $(C_COMMON_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $(OSHM_MOD_CINC) $< -o $@

$(OSHM_DIR)/obj/%.o: $(ROOT_DIR)/%.cpp
	$(OSHM_CPP) $(C_COMMON_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $(OSHM_MOD_CINC) $< -o $@

include $(wildcard $(OSHM_DIR)/obj/*.d)

$(OSHM_MOD_CLEAN):
	$(RM) $(RM_FLAGS) $(OSHM_MOD_POBJ)
	$(RM) $(RM_FLAGS) $(OSHM_MOD_PTARGET)
	$(RM) $(RM_FLAGS) $(OSHM_DIR)/obj/*.d
