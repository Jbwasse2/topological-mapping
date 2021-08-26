#!/usr/bin/env bash
## declare an array variable
#declare -a arr=("0.5" "0.75" "1.0" "1.25" "1.5" "2.0" "3.0" "4.0" "5.0")
declare -a arr=("0.0" "1.5" "3.0" "4.0" "5.0")
declare -a arr2=("5" "7" "10" "15" "20" "25" "30")
## now loop through the above array
mkdir ./results/VO50_experiment
cp sim_model.py sim_model_VO.py
sed -i "s|map_type_test =.*|map_type_test = 'VO'|g" sim_model_VO.py
for j in "${arr2[@]}"
do
    for i in "${arr[@]}"
    do
        sed -i "s|closeness=.*|closeness= $i|g" sim_model_VO.py
        sed -i "s|closeness =.*|closeness = $i|g" sim_model_VO.py
        sed -i "s|MAX_NUMBER_OF_STEPS =.*|MAX_NUMBER_OF_STEPS = $j|g" sim_model_VO.py
        filename=$(echo "$i"_"$j")
        echo $filename
        python sim_model_VO.py > ./results/VO50_experiment/$filename 2>&1
    done
done
rm sim_model_VO.py
