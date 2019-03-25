time mpirun -np 4 --hostfile host2 -mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 circuitcuda -l 10 -p 200 -n 100 -w 150 -b 16 -t 256

