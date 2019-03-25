time mpirun -np 36 --hostfile host5_36 --mca btl self,sm,tcp mm_mpicuda -m /home/rgu/workspace/ben_suit/data_set/dset9972.nn -b 64 -t 128 >nps_result/mpi_nld_5_36_10k.txt
