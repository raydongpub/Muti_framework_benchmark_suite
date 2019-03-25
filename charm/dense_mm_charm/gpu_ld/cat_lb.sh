echo charm_lb
for i in 5
do
  for j in 20 36 49 64 
     do
      echo nlb_$(j)
      cat lb_result/charm-mm_lb_${i}_${j}.out |grep -c "nps1"
      cat lb_result/charm-mm_lb_${i}_${j}.out |grep -c "nps2"
      cat lb_result/charm-mm_lb_${i}_${j}.out |grep -c "nps4"
      cat lb_result/charm-mm_lb_${i}_${j}.out |grep -c "nps5"
  done
done
echo charm_nlb
for i in 5
do
  for j in 20 36 49 64 
     do
      echo lb_$(j)
      cat lb_result/charm-mm_lb_${i}_${j}.out |grep -c "nps1"
      cat lb_result/charm-mm_lb_${i}_${j}.out |grep -c "nps2"
      cat lb_result/charm-mm_lb_${i}_${j}.out |grep -c "nps4"
      cat lb_result/charm-mm_lb_${i}_${j}.out |grep -c "nps5"
  done
done
