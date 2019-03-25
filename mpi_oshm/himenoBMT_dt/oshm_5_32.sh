time mpirun -np 32 --hostfile host5_32 -mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 bmt_oshmcuda -pe 4 4 2 -ds xl -b 16 -t 64
