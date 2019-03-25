time mpirun -np 49 --hostfile host5_49 --mca btl self,sm,tcp mm_mpicuda -m /home/rgu/workspace/ben_suit/data_set/dset10045.nn -b 64 -t 128 >nps_result/mpi_nld_5_49_10k.txt
