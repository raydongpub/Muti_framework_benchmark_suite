#srun -p gpu3 -N1 make clean;srun -p gpu3 -N1 make
#make clean;make -j4
sh charm_5_64.sh> nlb_result/charm-nw_nlb_5_64.out
sh charmlb_5_64.sh> lb_result/charm-nw_lb_5_64.out
