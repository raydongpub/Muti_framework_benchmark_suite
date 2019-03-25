#CHARM_DIR = /home/ruidong/Documents/software/charm/examples/charm++/net-linux-x86_64
CHARM_DIR = /usr/local/charm/mpi-linux-x86_64-smp
CHARM_INC = -I$(CHARM_DIR)/include
CHARM_LIB = -L$(CHARM_DIR)/lib
CC_FLAG   = -language charm++

CHARMC = charmc

BINARY = test

all: $(BINARY)
$(BINARY): test.o
	$(CHARMC) $(CHARM_INC) $(CC_FLAG) -o $@ $< $(CHARM_LIB)

test.o: test.C test.decl.h
	$(CHARMC) -c test.C
test.decl.h: test.ci
	$(CHARMC) $<

clean:
	rm -f *.o *decl.h *def.h $(BINARY) charmrun
