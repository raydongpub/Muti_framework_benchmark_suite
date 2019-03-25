time mpirun -np 32 --hostfile host5_32 --mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 nbody -c mpi.conf -s 32 +balancer RefineLB >nps_result/charm_ld_5_32_1200k.txt

