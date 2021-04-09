#!/usr/bin/env bash

last_scene="Azusa"
new_scene="Bolton"
#First perform navigation experiment on normal map
#cd rl
#echo "Results_Topological"
#sed -i "s/$last_scene/$new_scene/g" sim_model.py
#cd ..
#Clean Map
#Get pose
cd map
sed -i "s|env = .*|env = '$new_scene'|g" make_map.py_wor
sed -i "s|G = main.*|G = main(env, map_type='pose')|g" make_map_worm_experiment.py
sed -i "s|G = main.*|G = main(env, map_type='topological')|g" make_map_worm_experiment.py
