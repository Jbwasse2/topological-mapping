#!/usr/bin/env bash
declare -a arr=("0.5" "0.6" "0.7" "0.8" "0.85" "0.9" "0.95" "0.97" "0.98" "0.99")
declare -a arr2=("2.0" "3.0" "4.0" "5.0")

mkdir ./results/topExperiment
## now loop through the above array
for j in "${arr2[@]}"
do
    for i in "${arr[@]}"
    do
        sed -i "s|test_similarityEdges =.*|test_similarityEdges = $i|g" make_map_worm_experiment.py
        sed -i "s|closeness=.*|closeness= $j|g" make_map_worm_experiment.py
        sed -i "s|closeness =.*|closeness = $j|g" make_map_worm_experiment.py
        filename=$(echo "$i"_"$j")
        python make_map_worm_experiment.py > ./results/topExperiment/$filename 2>&1
    done
done
