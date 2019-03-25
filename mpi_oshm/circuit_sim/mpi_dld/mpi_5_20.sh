time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_20 circuitcuda -l 20 -p 2400 -n 100 -w 150 -pp 4 -m 20 -b 64 -t 512 >nps_result/mpi_ld_5_20_l20p2400n100w150.txt  
