# from util import bag_wrapper, play_rosbag, run_node
import top_map.extractor  # noqa
import top_map.util  # noqa
import top_map.waypoint  # noqa
import pathlib

import top_map.top_data

pathlib.Path("./output/wp").mkdir(parents=True, exist_ok=True)
