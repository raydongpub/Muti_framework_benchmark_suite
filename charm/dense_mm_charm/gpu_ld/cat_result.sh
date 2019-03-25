echo charm_lb
for i in 5
do
  for j in 20 36 49 64 
     do
      cat lb_result/charm-mm_lb_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
  done
done
echo charm_nlb
for i in 5
do
  for j in 20 36 49 64 
     do
      cat nlb_result/charm-mm_nlb_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
  done
done
