#!/usr/bin/env bash
declare -a arr=("0.5" "0.6" "0.7" "0.8" "0.85" "0.9" "0.95" "0.97" "0.98" "0.99")
gpu=1

## now loop through the above array
mkdir -p ./results/sim50experiment/
cp make_map_worm_experiment.py make_map_worm_experimentSIM.py
sed -i "s|map_type_test =.*|map_type_test = 'similarity'|g" make_map_worm_experimentSIM.py
sed -i "s|set_GPU =.*|set_GPU = '$gpu'|g" make_map_worm_experimentSIM.py
sed -i "s|env =.*|env = 'Browntown'|g" make_map_worm_experimentSIM.py
for i in "${arr[@]}"
do
    echo "$i"
    sed -i "s|test_similarityEdges =.*|test_similarityEdges = $i|g" make_map_worm_experimentSIM.py
    python make_map_worm_experimentSIM.py > ./results/sim50experiment/$i 2>&1
done
rm make_map_worm_experimentSIM.py
