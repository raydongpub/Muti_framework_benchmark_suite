time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_64 vecmmcuda -pe 8 -pn 5 -m 9984 -pp 4 >nps_result/mpi_ld_5_64_10k.txt
