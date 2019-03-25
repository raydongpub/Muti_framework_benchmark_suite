time mpirun -np 4 --hostfile host2 -mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 nw-single-stream -ds 2 -l 100 -n 400 -b 64 -t 512
