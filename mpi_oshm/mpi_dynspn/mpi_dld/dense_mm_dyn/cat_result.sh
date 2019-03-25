echo mpi_nps
for i in 5
do
  for j in 16 36 49 64 
     do
      cat nps_result/mpi-mm_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
  done
done
#echo oshm
#for i in 5
#do
#  for j in 16 36 49 64 
#     do
#      cat nps_result/oshm-mm_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
#  done
#done
