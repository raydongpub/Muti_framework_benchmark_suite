#time mpirun -np 20 --hostfile host5_20 -mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 mm_mpicuda -m /home/rgu/workspace/ben_suit/data_set/dset10k.nn -b 64 -t 128
time mpirun -np 20 --oversubscribe --hostfile host5_20 mm_mpicuda -m /home/rgu/workspace/ben_suit/data_set/dset10k.nn -b 64 -t 128 >nps_result/mpi_nld_test_10k.txt
#time mpirun -np 4 --oversubscribe mm_mpicuda -m /home/rgu/workspace/ben_suit/data_set/dset16.nn -b 64 -t 128 
