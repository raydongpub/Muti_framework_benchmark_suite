time mpirun -np 4 --hostfile host2 --mca btl self,sm,tcp  mm_mpicuda -m /home/rgu/workspace/ben_suit/data_set/dset16.nn -b 64 -t 128
