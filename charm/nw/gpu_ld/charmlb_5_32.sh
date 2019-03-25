time mpirun -np 32 --hostfile host5_32 nw -l 1000 -n 4992 -f 4 -s 32 -b 64 -t 512 +balancer GreedyLB >nps_result/charm_ld_5_32_l1000n5000.txt
