echo MPI
for i in 2 4 9
do
  for j in 20 32 48 64 
     do
      cat scala_result/mpi/mpi-himeno_${i}_${j}.o* |grep "and time:"|awk -v max=0 '{if($8>max){want=$8; max=$8}}END{print want}'
  done
done
echo OSHM
for i in 2 4 9
do
  for j in 20 32 48 64 
     do
      cat scala_result/oshm/oshm-himeno_${i}_${j}.o* |grep "and time:"|awk -v max=0 '{if($8>max){want=$8; max=$8}}END{print want}'
  done
done
