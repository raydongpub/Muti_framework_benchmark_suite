echo charm_lb
for i in 5
do
  for j in 20 32 48 64 
     do
      cat nps_result/charm_ld_${i}_${j}_l1000n5000.txt |grep "Time"|awk '{print $2}'
  done
done
      #cat nps_result/charm_ld_${i}_${j}_l1000n5000.txt |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
#echo charm_nlb
#for i in 5
#do
#  or j in 20 32 48 64 
#     do
#      cat nlb_result/charm-nw_nlb_${i}_${j}.out |grep "Time:"|awk -v max=0 '{if($2>max){want=$2; max=$2}}END{print want}'
#  done
#done
