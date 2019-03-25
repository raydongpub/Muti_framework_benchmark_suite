time mpirun -np 48 --hostfile host5_48 -mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 bmt_oshmcuda -pe 4 4 3 -ds xl -b 16 -t 64
