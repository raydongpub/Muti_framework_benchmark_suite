time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_32 himenocuda -pe 4 4 2 -ds xl -pn 5 -pm 32 -pp 4 -it 20 >nps_result/mpi_ld_5_32_xl.txt

