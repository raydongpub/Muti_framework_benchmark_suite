#srun -p gpu3 -N1 make clean;srun -p gpu3 -N1 make
#make clean;make -j4
sh mpi_5_20.sh> nps_result/mpi-mm_5_20.out
sh oshm_5_20.sh> nps_result/oshm-mm_5_20.out
#sh mpilb_5_64.sh> lb_result/mpi-mm_lb_5_64.out
