include ../common/make.config

all: nw-single-stream

global.o: global.c
	$(GCC) $(GCC_FLAGS) -I$(INCLUDE) $^ -c -o $@

nw_gpu.o: nw_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE) $^ -c -o $@

nw_cpu.o: nw_cpu.c
	$(GCC) $(GCC_FLAGS) $^ -c -o $@

nw-multi-stream.o: nw-multi-stream.cu
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE) $(MPI) $< -c -o $@

nw-single-stream.o: nw-single-stream.cu
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE) $(MPI) $< -c -o $@

nw-multi-stream:	nw-multi-stream.o global.o nw_cpu.o nw_gpu.o
	$(NVCC) $(NVCC_FLAGS) $(LIB_P) $^ -o $@ $(LIB_S)

nw-single-stream:	nw-single-stream.o global.o nw_gpu.o
	$(NVCC) $(NVCC_FLAGS) $(LIB_P) $^ -o $@ $(LIB_S)

clean:
	rm -f *.o nw-multi-stream nw-single-stream
