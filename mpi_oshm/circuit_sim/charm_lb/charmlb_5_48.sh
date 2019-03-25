time mpirun -np 48 --hostfile host5_48 --mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 circuit_sim 2400 100 150 20 48 +balancer RefineLB >nps_result/charm_ld_5_48_l50p2400n100w150.txt
