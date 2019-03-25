time mpirun -np 20 --hostfile host5_20 -mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 circuitcuda -l 20 -p 2400 -n 100 -w 150 -b 16 -t 256 
