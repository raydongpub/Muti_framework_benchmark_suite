time mpirun -np 3 --oversubscribe --map-by node --hostfile host2 -mca oobtcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 nbodycuda -c mpi12k.conf -pe 4 -pn 3 -pp 2
