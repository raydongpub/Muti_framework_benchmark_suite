time mpirun -np 64 --hostfile host5_64 --mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 circuit_sim 2464 100 150 20 64 +balancer RefineLB >nps_result/charm_ld_5_64_l50p2400n100w150.txt
