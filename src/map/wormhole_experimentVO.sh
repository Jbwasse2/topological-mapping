#!/usr/bin/env bash
#declare -a arr=("0.0" "0.5" "0.75" "1.0" "1.25" "1.5" "2.0" "3.0" "4.0" "5.0")
declare -a arr=("0.0" "1.5")
gpu=0

## now loop through the above array
mkdir -p ./results/VO50experiment/
cp make_map_worm_experiment.py make_map_worm_experimentVO.py
sed -i "s|map_type_test =.*|map_type_test = 'VO'|g" make_map_worm_experimentVO.py
sed -i "s|set_GPU =.*|set_GPU = '$gpu'|g" make_map_worm_experimentVO.py
sed -i "s|env =.*|env = 'Browntown'|g" make_map_worm_experimentVO.py
for i in "${arr[@]}"
do
    echo "$i"
    sed -i "s|test_closeness=.*|test_closeness= $i|g" make_map_worm_experimentVO.py
    sed -i "s|test_closeness =.*|test_closeness = $i|g" make_map_worm_experimentVO.py
    python make_map_worm_experimentVO.py > ./results/VO50experiment/$i 2>&1
done
rm make_map_worm_experimentVO.py
