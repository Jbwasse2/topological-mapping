import os
import pathlib
import subprocess

import pudb

pathlib.Path("./test/results/wp/").mkdir(parents=True, exist_ok=True)
pathlib.Path("./test/results/planner/").mkdir(parents=True, exist_ok=True)
pathlib.Path("./data/indoorData/results/top_maps/").mkdir(parents=True,
                                                          exist_ok=True)
if not os.path.exists("./test/testing_resources/rosbag/test.bag"):
    raise OSError(
        "Missing test.bag. Please download testing resources from GitHub!")
# Check to make sure roscore and bridge is running
command1 = "export PYTHONPATH='' && "
command2 = ". /opt/ros/melodic/setup.sh && "
command3 = "rosnode list"
command = command1 + command2 + command3
process = subprocess.Popen(command,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           shell=True)
out, err = process.communicate()
out = out.splitlines()
flag_roscore = 0
flag_bridge = 0
for line in out:
    string_line = line.decode("utf-8")
    if 'rosout' in string_line:
        flag_roscore = 1
    if 'ros_bridge' in string_line:
        flag_bridge = 1
# if not flag_roscore:
#    raise Exception("Please run ROSCORE!")
# if not flag_bridge:
#    raise Exception("Please run ros1/2 Bridge!")
