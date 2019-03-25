time mpirun -np 20 --hostfile host5_20 -mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 bmt_oshmcuda -pe 5 2 2 -ds xl -b 16 -t 64
