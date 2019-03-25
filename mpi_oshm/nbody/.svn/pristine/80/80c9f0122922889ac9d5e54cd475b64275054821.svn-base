
.PHONY: $(IVM_CUDA_MOD_CLEAN)

$(IVM_CUDA_MOD_PTARGET) : $(IVM_CUDA_MOD_POBJ)
	$(NVCC_LD) $(LD_IVM_PFLAGS) $^ -o $@ $(LD_IVM_CUDA_SFLAGS)
	@printf "\n***** IVM_CUDA modules built successfully *****\n\n"

$(IVM_CUDA_DIR)/obj/%.o : $(IVM_CUDA_DIR)/%.cpp
	$(CPP) $(C_IVM_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $(C_DEBUG_FLAGS) $(IVM_CUDA_MOD_CINC)  $< -o $@

$(IVM_CUDA_DIR)/obj/%.o : $(IVM_CUDA_DIR)/%.cu
	$(NVCC) $(C_IVM_FLAGS) $(C_DEBUG_FLAGS) $(IVM_CUDA_MOD_CINC) $(IVM_CUDA_MOD_NVCCFLAGS) $< -o $@

$(IVM_CUDA_DIR)/obj/%.o : $(ROOT_DIR)/%.cpp
	$(CPP) $(C_IVM_FLAGS) $(C_WARNING_FLAGS) $(C_SHARED_FLAGS) $(C_DEP_FLAGS) $(IVM_CUDA_MOD_CINC) $< -o $@

include $(wildcard $(IVM_CUDA_DIR)/obj/*.d)

$(IVM_CUDA_MOD_CLEAN) :
	$(RM) $(RM_FLAGS) $(IVM_CUDA_MOD_POBJ) 
	$(RM) $(RM_FLAGS) $(IVM_CUDA_MOD_PTARGET)
	$(RM) $(RM_FLAGS) $(IVM_CUDA_DIR)/obj/*.d

