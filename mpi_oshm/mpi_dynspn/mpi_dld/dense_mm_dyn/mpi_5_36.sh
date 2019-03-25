time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_36 vecmmcuda -pe 6 -pn 5 -m 9972 -pp 4 >nps_result/mpi_ld_5_36_10k.txt
