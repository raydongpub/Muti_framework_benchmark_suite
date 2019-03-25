time mpirun -np 48 --hostfile host5_48 --mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 nbody -c mpi.conf -s 48 +balancer RefineLB >nps_result/charm_ld_5_48_1200k.txt

