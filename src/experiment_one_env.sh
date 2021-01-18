#!/usr/bin/env bash
env=$1
if [ -z "$env" ]
then
    echo "No env given"
    exit 1
fi
#First collect the data for the experiment in simulation (This gets the best traj)
#Replace generate file with current env and then get dataset
#sed -i "s|_generate_fn('.*|_generate_fn('./data/scene_datasets/gibson/"$env".glb')|g" ./habitat-lab/habitat_baselines/rl/ddppo/data_generation/create_gibson_large_dataset.py
#python ./habitat-lab/habitat_baselines/rl/ddppo/data_generation/create_gibson_large_dataset.py
#Next get the images for the collected trajs
yes | rm -rf ./data/datasets/pointnav/gibson/v4/train_large/images/$env
python ./data_collection/get_train_large_data.py -e $env
