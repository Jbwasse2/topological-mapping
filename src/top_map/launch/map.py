from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    map_node = Node(
        package="top_map",
        node_executable="mapper",
        parameters=[
            {
                "use_pose_estimate": True,
                "close_distance": 1.0,
                "confidence": 0.85,
                "save_meng": True,
            }
        ],
    )
    ld.add_action(map_node)
    return ld
