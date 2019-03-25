echo mpi_nps
for i in 5
do
  for j in 20 32 48 64 
     do
      cat nps_result/mpi-cct_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
  done
done
#echo mpi_nnps
#for i in 5
#do
#  or j in 20 32 48 64 
#     do
#      cat nnps_result/mpi-cct_nnps_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
#  done
#done
