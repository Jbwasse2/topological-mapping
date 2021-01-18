#!/usr/bin/env bash
env=$1
if [ -z "$env" ]
then
    echo "No env given"
    exit 1
fi
trajs=$2
if [ -z "$trajs" ]
then
    echo "No Number of Trajectories to Collect Given"
fi
ip=$3

#First collect the data for the experiment in simulation (This gets the best traj)
#Replace generate file with current env and then get dataset
sed -i "s|_generate_fn('.*|_generate_fn('./data/scene_datasets/gibson/"$env".glb')|g" ./habitat-lab/habitat_baselines/rl/ddppo/data_generation/create_gibson_large_dataset.py
sed -i "s|NUM_EPISODES_PER_SCENE = 2|NUM_EPISODES_PER_SCENE = $trajs|g" ./habitat-lab/habitat_baselines/rl/ddppo/data_generation/create_gibson_large_dataset.py
python ./habitat-lab/habitat_baselines/rl/ddppo/data_generation/create_gibson_large_dataset.py
#Next get the images for the collected trajs
yes | rm -rf ./data/datasets/pointnav/gibson/v4/train_large/images/$env
python ./data_collection/get_train_large_data.py -e $env
#Visualization
if [ ! -z "$ip" ]
then
    ffmpeg -framerate 60 -pattern_type glob -i './data/datasets/pointnav/gibson/v4/train_large/images/'$env'/*.jpg' -c:v libx264 -r 60 -pix_fmt yuv420p out_rgb.mp4
fi
#Get pose labels of collected trajectory
cd slam
rm out/*
sed -i "s|NUMBER_OF_TRAJECTORIES_COLLECTED =.*|NUMBER_OF_TRAJECTORIES_COLLECTED = $trajs|g" main.py
sed -i "s|main('.*|main('$env')|g" main.py
python main.py
cd out
#Get slam visualization (Top down map)
if [ ! -z "$ip" ]
then
    ffmpeg -framerate 60 -pattern_type glob -i '*.png' -c:v libx264 -r 60 -pix_fmt yuv420p ../../out_slam.mp4
fi
cd ../..
#Combine RGB and top down and send over
if [ ! -z "$ip" ]
then
    ffmpeg -i out_slam.mp4 -s 256x256 -c:a copy out_slam_256.mp4
    new_rgb_duration=$(ffmpeg -i out_slam_256.mp4 2>&1 | grep Duration | cut -d ' ' -f 4 | sed s/,//)
    ffmpeg -i out_rgb.mp4 -t $new_rgb_duration out_rgb_short.mp4
    ffmpeg -i out_rgb_short.mp4 -i out_slam_256.mp4 -filter_complex hstack=inputs=2 send_me_over.mp4
    rsync -avzh send_me_over.mp4 justin@$ip:/home/justin/Downloads/
    rm ./*.mp4
fi
#Build topological map
