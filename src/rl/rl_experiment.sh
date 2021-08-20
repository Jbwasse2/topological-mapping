#!/usr/bin/env bash
## declare an array variable
declare -a arr2=("5" "7" "10" "15" "20" "25" "30" "40" "50")
#declare -a arr=("0.5" "0.6" "0.7" "0.8" "0.85" "0.9" "0.95" "0.97" "0.98" "0.99")
declare -a arr=("0.5")

## now loop through the above array
for i in "${arr2[@]}"
do
    echo "$i"
    mkdir ./results/$i
    sed -i "s|MAX_NUMBER_OF_STEPS =.*|MAX_NUMBER_OF_STEPS = $i|g" sim_model.py
   # or do whatever with individual element of the array
    for p in "${arr[@]}"
    do
        echo "*******"
        echo "$p"
        echo "*******"
        #sed -i "s|similarity_test =.*|similarity_test = '$p'|g" sim_model.py
        #sed -i "s|map_type_test =.*|map_type_test = '$p'|g" sim_model.py
        python sim_model.py > ./results/clean/$p 2>&1
    done
    mv ./results/clean/* ./results/$i
done
