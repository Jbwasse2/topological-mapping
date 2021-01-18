#!/usr/bin/env bash
#Note: To get initial points in train_large dataset look at ddppoo data colleciton in habitat_baseline
for f in ../../data/datasets/pointnav/gibson/v4/train_large/content/*; do
    env_name="$(basename -- $f | cut -f 1 -d '.')"
    echo "$env_name"
    python get_train_large_data.py -e "$env_name"
done
