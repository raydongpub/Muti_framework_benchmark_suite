time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_48 -mca oobtcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 nwcuda -ds 2 -l 1000 -n 4992 -b 64 -t 512 -pe 48 -pn 5 -pp 4 -div 4 >nps_result/mpi_ld_5_48_l1000n5000.txt
