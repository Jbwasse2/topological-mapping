#!/usr/bin/env bash
## declare an array variable
#declare -a arr=("0.5" "0.75" "1.0" "1.25" "1.5" "2.0" "3.0" "4.0" "5.0")
## now loop through the above array
declare -a arr2=("5" "7" "10" "15" "20" "25" "30")
declare -a arr=("0.0" "0.5" "0.6" "0.7" "0.8" "0.85" "0.9" "0.95" "0.97" "0.98" "0.99")
mkdir ./results/SIM50_experiment
cp sim_model.py sim_model_SIM.py
sed -i "s|map_type_test =.*|map_type_test = 'similarity'|g" sim_model_SIM.py
for j in "${arr2[@]}"
do
    for i in "${arr[@]}"
    do
        sed -i "s|similarity_test =.*|similarity_test = '$i'|g" sim_model_SIM.py
        sed -i "s|MAX_NUMBER_OF_STEPS =.*|MAX_NUMBER_OF_STEPS = $j|g" sim_model_SIM.py
        filename=$(echo "$i"_"$j")
        echo $filename
        python sim_model_SIM.py > ./results/SIM50_experiment/$filename 2>&1
    done
done
rm sim_modely_SIM.py
