#!/usr/bin/env bash
last_env="Holcut"
while read p; do
    echo "*******"
    echo "$last_env"
    echo "$p"
    echo "*******"
    sed -i "s|$last_env|$p|g" sim_model.py
    python sim_model.py > ./results/clean/$p 2>&1
    last_env=$p
done < ./test_envs.txt
