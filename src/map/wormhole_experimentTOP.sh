#!/usr/bin/env bash
declare -a arr=("0.5" "0.6" "0.7" "0.8" "0.85" "0.9" "0.95" "0.97" "0.98" "0.99")
declare -a arr2=("2.0" "3.0" "4.0" "5.0")

mkdir -p ./results/top50Experiment
cp make_map_worm_experiment.py make_map_worm_experimentTOP.py
## now loop through the above array
sed -i "s|map_type_test =.*|map_type_test = 'topological'|g" make_map_worm_experimentTOP.py
for j in "${arr2[@]}"
do
    for i in "${arr[@]}"
    do
        echo $i
        echo $j
        sed -i "s|test_similarityEdges =.*|test_similarityEdges = $i|g" make_map_worm_experimentTOP.py
        sed -i "s|test_closeness=.*|test_closeness= $j|g" make_map_worm_experimentTOP.py
        sed -i "s|test_closeness =.*|test_closeness = $j|g" make_map_worm_experimentTOP.py
        filename=$(echo "$i"_"$j")
        python make_map_worm_experimentTOP.py > ./results/top50Experiment/$filename 2>&1
    done
done
