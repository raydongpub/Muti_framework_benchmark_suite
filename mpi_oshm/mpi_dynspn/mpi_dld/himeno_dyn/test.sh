time mpirun -np 3 --oversubscribe --map-by node --hostfile host2 -mca oobtcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 himenocuda -pe 2 2 1 -ds s -pn 3 -pm 4 -pp 2
