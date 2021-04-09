#!/usr/bin/env bash
cd map
for i in $(seq 1 20); do
    for j in $(seq 1 20); do
        cd map
        sed -i "s|sparsity=.*|sparsity=$i,|g" make_map.py
        sed -i "s|similarity=.*|similarity=$j,|g" make_map.py
        touch results/${i}_${j}
        python make_map.py >> ./results/${i}_${j} 2>&1
        cd ..
        cd rl
        touch results/${i}_${j}
        python sim_model.py >> ./results/${i}_${j} 2>&1
        cd ..
    done
done
