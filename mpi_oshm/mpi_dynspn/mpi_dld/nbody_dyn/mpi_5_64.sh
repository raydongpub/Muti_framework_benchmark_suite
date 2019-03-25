time mpirun -np 5 --oversubscribe --map-by node --hostfile host5_64 nbodycuda -c mpi1200k.conf -pe 64 -pn 5 -pp 4 >nps_result/mpi_ld_5_64_1200k.txt 
