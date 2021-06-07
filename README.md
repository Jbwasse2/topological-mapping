# Topological Mapping on Real World Robot
Note: Currently I develope in ROS2 Dashing, I can not gaurantee this code will work for other ROS2 versions.

# Setup
First you should create a new conda env
- `conda create -n topmap python=3.6`
- `conda activate topmap`

Now to get the code running  
- Clone this repo `git clone https://github.com/Jbwasse2/topological-mapping.git`
- Clone https://github.com/xymeng/rmp_nav
- `source ./set_envs.sh` in the rmp nav repo
- Source ROS2 `source /opt/ros/ROSVERSION/setup.sh` if you are using bash
- Source ROS2 `source /opt/ros/ROSVERSION/setup.zsh` if you are using zsh (like me)
- Install python packages `pip install -r requirements.txt`
- `cd ./topological-mapping/src`
- `colcon build`
- `source ./install/setup.zsh`
- sudo apt-get install ros-$(rosversion -d)-cv-bridge

When you want to rerun this code again at a later time, you will just need to source the RMP_NAV repo, ROS2, and the install/setup.zsh.

## Running Test
- `cd top_map`
- Run tests `python -m pytest -s test/`

## Extras
- **ORBSLAM2** Build and install orbslam2 (https://github.com/raulmur/ORB_SLAM2) and python bindings (https://github.com/jskinn/ORB_SLAM2-PythonBindings), [I found this script to be useful](https://github.com/facebookresearch/habitat-lab/blob/master/habitat_baselines/slambased/install_deps.sh). 

# Architecture
At building time
trajectory collection -> similarity detector -> sparse trajectory collection -> similarity detector -> topological map

At run time
current view -> nav policy -> action -> if reached local goal, update goal -> repeat until at final goal

## Similarity Detector
Given two images, what is the distance between the two images. Another metric that is often used is ?what is the probability the two images are close?"
Working Implementations:  
TODO:  
Meng et al  
Mine trained on real world data  

## Image - Nav Policy
Given the current view and a (local) goal image, predict an action in order to reach the goal  
Working Implementations:  
Meng et al  
TODO:  
