#!/usr/bin/env bash
## declare an array variable
#declare -a arr=("0.5" "0.75" "1.0" "1.25" "1.5" "2.0" "3.0" "4.0" "5.0")
## now loop through the above array
declare -a arr=("0.5" "0.6" "0.7" "0.8" "0.85" "0.9" "0.95" "0.97" "0.98" "0.99")
#declare -a arr2=("1.5" "2.0" "3.0" "4.0" "5.0")
declare -a arr2=("2.0" "3.0")
mkdir ./results/TOP_experiment
sed -i "s|map_type_test =.*|map_type_test = 'topological'|g" sim_model.py
for j in "${arr2[@]}"
do
    for i in "${arr[@]}"
    do
        sed -i "s|similarity_test =.*|similarity_test = '$i'|g" sim_model.py
        sed -i "s|closeness=.*|closeness= $j|g" sim_model.py
        sed -i "s|closeness =.*|closeness = $j|g" sim_model.py
        filename=$(echo "$i"_"$j")
        echo $filename
        python sim_model.py > ./results/TOP_experiment/$filename 2>&1
    done
done
