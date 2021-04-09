#!/usr/bin/env bash
last_env="Pasatiempo"
while read p; do
    echo "*******"
    echo "$last_env"
    echo "$p"
    echo "*******"
    sed -i "s|$last_env|$p|g" make_map.py
    python make_map.py > ./results/mapClean/$p 2>&1
    last_env=$p
done < ./worm_model/envs.txt
