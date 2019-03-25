$(DATASET_PTARGET): $(DATASET_POBJ)
	$(CPP) $(LD_PFLAGS) $^ -o $@ $(LD_SFLAGS)

$(SHOWPAR_PTARGET): $(SHOWPAR_POBJ)
	$(CPP) $(LD_PFLAGS) $^ -o $@ $(LD_SFLAGS)

$(ROOT_DIR)/obj/%.o: $(ROOT_DIR)/%.cpp
	$(CPP) $(C_COMMON_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $< -o $@

.PHONY: $(COMMON_CLEAN)

$(COMMON_CLEAN):
	$(RM) $(RM_FLAGS) $(DATASET_PTARGET)
	$(RM) $(RM_FLAGS) $(SHOWPAR_PTARGET)
	$(RM) $(RM_FLAGS) $(DATASET_POBJ)
	$(RM) $(RM_FLAGS) $(SHOWPAR_POBJ)
	$(RM) $(RM_FLAGS) $(ROOT_DIR)/obj/*.d

