time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_48 himenocuda -pe 4 4 3 -ds xl -pn 5 -pm 48 -pp 4 -it 20 >nps_result/mpi_ld_5_48_xl.txt

