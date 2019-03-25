echo charm_lb
for i in 5
do
  for j in 20 32 48 64 
     do
      echo lb_$(j)
      cat lb_result/charm-himeno_lb_${i}_${j}.out |grep -c "nps1"
      cat lb_result/charm-himeno_lb_${i}_${j}.out |grep -c "nps2"
      cat lb_result/charm-himeno_lb_${i}_${j}.out |grep -c "nps4"
      cat lb_result/charm-himeno_lb_${i}_${j}.out |grep -c "nps5"
  done
done
echo charm_nlb
for i in 5
do
  for j in 20 32 48 64 
     do
      echo nlb_$(j)
      cat nlb_result/charm-himeno_nlb_${i}_${j}.out |grep -c "nps1"
      cat nlb_result/charm-himeno_nlb_${i}_${j}.out |grep -c "nps2"
      cat nlb_result/charm-himeno_nlb_${i}_${j}.out |grep -c "nps4"
      cat nlb_result/charm-himeno_nlb_${i}_${j}.out |grep -c "nps5"
  done
done
