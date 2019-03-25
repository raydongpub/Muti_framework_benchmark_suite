time mpirun -np 32 --hostfile host5_32 -mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 nb_mpicuda -c mpi1200k.conf -b 16 -t 128 >nps_result/mpi_nld_5_32_1200k.txt
