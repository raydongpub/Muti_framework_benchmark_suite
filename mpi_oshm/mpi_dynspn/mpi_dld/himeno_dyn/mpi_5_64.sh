time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_64 himenocuda -pe 4 4 4 -ds xl -pn 5 -pm 64 -pp 4 -it 20 >nps_result/mpi_ld_5_64_xl.txt

