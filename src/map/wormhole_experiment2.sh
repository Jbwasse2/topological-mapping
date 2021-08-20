#!/usr/bin/env bash
declare -a arr=("0.5" "0.75" "1.0" "1.25" "1.5" "2.0" "3.0" "4.0" "5.0")

## now loop through the above array
for i in "${arr[@]}"
do
    echo "$i"
    sed -i "s|closeness=.*|closeness= $i|g" make_map_worm_experiment.py
    sed -i "s|closeness =.*|closeness = $i|g" make_map_worm_experiment.py
    python make_map_worm_experiment.py > ./results/VOexperiment/$i 2>&1
done
