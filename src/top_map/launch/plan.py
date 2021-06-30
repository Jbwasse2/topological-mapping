from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    planner_node = Node(
        package="top_map",
        node_executable="planner",
        parameters=[
            {"topological_map_pkl": "./test/testing_resources/test_top_map.pkl"}
        ],
    )
    waypoint_node = Node(
        package="top_map",
        node_executable="waypoint",
        parameters=[],
    )
    ld.add_action(planner_node)
    ld.add_action(waypoint_node)
    return ld
