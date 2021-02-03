#!/usr/bin/env bash
while read p; do
    echo "$p"
    sed -i "s|env =.*|env = '$p'|g" make_map.py
    python make_map.py > ./results/wormholesClean/$p 2>&1
done < ./worm_model/envs.txt
