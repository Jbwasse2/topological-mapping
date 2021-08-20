#!/usr/bin/env bash
declare -a arr=("0.5" "0.6" "0.7" "0.8" "0.85" "0.9" "0.95" "0.97" "0.98" "0.99")

## now loop through the above array
for i in "${arr[@]}"
do
    echo "$i"
    sed -i "s|test_similarityEdges =.*|test_similarityEdges = $i|g" make_map_worm_experiment.py
    python make_map_worm_experiment.py > ./results/simExperiment/$i 2>&1
done
