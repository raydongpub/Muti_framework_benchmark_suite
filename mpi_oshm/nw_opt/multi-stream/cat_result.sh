echo mpi_nps
for i in 5
do
  for j in 20 32 48 64 
     do
      cat nps_result/mpi_nld_${i}_${j}_l1000n5000.txt |grep "Time"|awk -v max=0 '{if($3>max){want=$3; max=$3}}END{print want}'
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
