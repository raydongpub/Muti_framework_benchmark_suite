echo mpi_2s
for i in 5
do
  for j in 16 36 49 64 
     do
      echo nps_$(j)
      cat nps_result/mpi-mm_${i}_${j}.out |grep -c "nps1"
      cat nps_result/mpi-mm_${i}_${j}.out |grep -c "nps2"
      cat nps_result/mpi-mm_${i}_${j}.out |grep -c "nps4"
      cat nps_result/mpi-mm_${i}_${j}.out |grep -c "nps5"
  done
done

#echo oshm
#for i in 5
#do
#  for j in 16 36 49 64 
#     do
#      echo nps_$(j)
#      cat nps_result/oshm-mm_${i}_${j}.out |grep -c "nps1"
#      cat nps_result/oshm-mm_${i}_${j}.out |grep -c "nps2"
#      cat nps_result/oshm-mm_${i}_${j}.out |grep -c "nps4"
#      cat nps_result/oshm-mm_${i}_${j}.out |grep -c "nps5"
#  done
#done
