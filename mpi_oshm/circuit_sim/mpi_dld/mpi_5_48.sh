time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_48 circuitcuda -l 20 -p 2400 -n 100 -w 150 -pp 4 -m 48 -b 64 -t 512 >nps_result/mpi_ld_5_48_l20p2400n100w150.txt 
