#!/usr/bin/env bash
## declare an array variable
declare -a arr=("25" "27" "30" "40" "50" "60" "70" "80" "90" "100")

## now loop through the above array
for i in "${arr[@]}"
do
    echo "$i"
    mkdir ./results/$i
    sed -i "s|MAX_NUMBER_OF_STEPS =.*|MAX_NUMBER_OF_STEPS = $i|g" sim_model.py
   # or do whatever with individual element of the array
    while read p; do
        echo "*******"
        echo "$p"
        echo "*******"
        sed -i "s|map_type_test =.*|map_type_test = '$p'|g" sim_model.py
         python sim_model.py > ./results/clean/$p 2>&1
    done < ./map_types.txt
    mv ./results/clean/* ./results/$i
done
