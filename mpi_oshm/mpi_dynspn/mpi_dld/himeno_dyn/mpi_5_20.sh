time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_20 himenocuda -pe 5 2 2 -ds xl -pn 5 -pm 20 -pp 4 -it 20 >nps_result/mpi_ld_5_20_xl.txt

