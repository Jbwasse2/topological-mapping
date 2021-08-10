#!/usr/bin/env bash
while read p; do
    echo "$p"
#    sed -i "s|env =.*|env = '$p'|g" make_map_worm_experiment.py
    sed -i "s|map_type_test =.*|map_type_test = '$p'|g" make_map_worm_experiment.py
    python make_map_worm_experiment.py > ./results/wormholes/$p 2>&1
done < ./map_types.txt
