echo mpi_2s
for i in 5
do
  for j in 20 32 48 64 
     do
      echo nps_$(j)
      cat nps_result/mpi-cct_${i}_${j}.out |grep -c "nps1"
      cat nps_result/mpi-cct_${i}_${j}.out |grep -c "nps2"
      cat nps_result/mpi-cct_${i}_${j}.out |grep -c "nps4"
      cat nps_result/mpi-cct_${i}_${j}.out |grep -c "nps5"
  done
done

#echo mpi_nnps
#for i in 5
#do
#  for j in 20 32 48 64 
#     do
#      echo nnps_$(j)
#      cat nnps_result/mpi-cct_nnps_${i}_${j}.out |grep -c "nps1"
#      cat nnps_result/mpi-cct_nnps_${i}_${j}.out |grep -c "nps2"
#      cat nnps_result/mpi-cct_nnps_${i}_${j}.out |grep -c "nps4"
#      cat nnps_result/mpi-cct_nnps_${i}_${j}.out |grep -c "nps5"
#  done
#done
