time mpirun -np 20 --hostfile host5_20 --mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 circuit_sim 2400 100 150 20 20 +balancer RefineLB >nps_result/charm_ld_5_20_l50p2400n100w150.txt
