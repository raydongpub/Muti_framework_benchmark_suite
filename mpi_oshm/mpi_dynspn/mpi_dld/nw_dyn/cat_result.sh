echo mpi_nps
for i in 5
do
  for j in 20 32 48 64 
     do
      cat nps_result/mpi-nw_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
  done
done
#echo oshm
#for i in 5
#do
#  for j in 20 32 48 64 
#     do
#      cat nps_result/oshm-nw_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
#  done
#done
