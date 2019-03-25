time mpirun -np 3 --oversubscribe --map-by node --hostfile host2 -mca oobtcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 nwcuda -ds 2 -l 200 -n 400 -b 64 -t 512 -pe 2 -pn 3 -pp 2 -div 2
