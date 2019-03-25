time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_49 vecmmcuda -pe 7 -pn 5 -m 10045 -pp 4 >nps_result/mpi_ld_5_49_10k.txt
