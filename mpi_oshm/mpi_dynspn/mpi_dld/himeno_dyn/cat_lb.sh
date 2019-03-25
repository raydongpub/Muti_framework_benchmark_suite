echo mpi_2s
for i in 5
do
  for j in 20 32 48 64 
     do
      echo nps_$(j)
      cat nps_result/mpi-himeno_${i}_${j}.out |grep -c "nps1"
      cat nps_result/mpi-himeno_${i}_${j}.out |grep -c "nps2"
      cat nps_result/mpi-himeno_${i}_${j}.out |grep -c "nps4"
      cat nps_result/mpi-himeno_${i}_${j}.out |grep -c "nps5"
  done
done

#echo oshm
#for i in 5
#do
#  for j in 20 32 48 64 
#     do
#      echo nps_$(j)
#      cat nps_result/oshm-himeno_${i}_${j}.out |grep -c "nps1"
#      cat nps_result/oshm-himeno_${i}_${j}.out |grep -c "nps2"
#      cat nps_result/oshm-himeno_${i}_${j}.out |grep -c "nps4"
#      cat nps_result/oshm-himeno_${i}_${j}.out |grep -c "nps5"
#  done
#done
