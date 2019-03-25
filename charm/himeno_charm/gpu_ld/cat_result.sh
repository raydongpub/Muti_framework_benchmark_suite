echo charm_lb
for i in 5
do
  for j in 20 32 48 64 
     do
      cat lb_result/charm-himeno_lb_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
  done
done
echo charm_nlb
for i in 5
do
  or j in 20 32 48 64 
     do
      cat nlb_result/charm-himeno_nlb_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
  done
done
