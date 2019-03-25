time mpirun -np 64 --hostfile host5_64 --mca btl self,sm,tcp mm_mpicuda -m /home/rgu/workspace/ben_suit/data_set/dset9984.nn -b 64 -t 128 >nps_result/mpi_nld_5_64_10k.txt
