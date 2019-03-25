time mpirun -np 20 --hostfile host5_20 -mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 nb_mpicuda -c mpi1200k.conf -b 16 -t 128 >nps_result/mpi_nld_5_20_1200k.txt
