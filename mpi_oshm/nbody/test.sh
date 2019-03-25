time mpirun -np 4 --hostfile host2 -mca oob_tcp_if_exclude virbr0 --mca btl_tcp_if_exclude virbr0 nb_mpicuda -c mpi12k.conf -b 64 -t 256

