time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_16 vecmmcuda -pe 4 -pn 5 -m 10000 -pp 4 >nps_result/mpi_ld_5_20_10k.txt
