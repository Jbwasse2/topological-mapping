#!/usr/bin/env bash
while read p; do
    sed -i "s|scene = .*|scene = '$p'|g" main.py
    python main.py
done < ../map/worm_model/envs.txt
