import os
import pathlib

pathlib.Path("./test/results/wp/").mkdir(parents=True, exist_ok=True)
pathlib.Path("./test/results/planner/").mkdir(parents=True, exist_ok=True)
pathlib.Path("./data/indoorData/results/top_maps/").mkdir(parents=True, exist_ok=True)
if not os.path.exists("./test/testing_resources/rosbag/test.bag"):
    raise OSError("Please download testing resources from GitHub!")
