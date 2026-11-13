"""Lance Thor + la policy : ros2_control (contrôleur de position à 30 Hz), RViz et le nœud thor_policy.

  Test sans matériel (TomLinux)  : ros2 launch thor_policy policy_robot.launch.py use_mock:=true
  Vrai bras (Dell, au kot)       : ros2 launch thor_policy policy_robot.launch.py max_excursion_deg:=15

Puis, dans un autre terminal :
  ros2 topic pub --once /thor_policy/target geometry_msgs/msg/PointStamped "{header: {frame_id: base_link}, point: {x: 0.0, y: 0.32, z: 0.22}}"
  ros2 service call /thor_policy/enable std_srvs/srv/SetBool "{data: true}"
  ros2 service call /thor_policy/enable std_srvs/srv/SetBool "{data: false}"   # arrêt
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_mock = LaunchConfiguration("use_mock")
    rviz = LaunchConfiguration("rviz")
    max_exc = LaunchConfiguration("max_excursion_deg")
    vmax_scale = LaunchConfiguration("vmax_scale")
    lag_stop = LaunchConfiguration("lag_stop_deg")

    share = get_package_share_directory("thor_policy")
    xacro_file = os.path.join(get_package_share_directory("thor_urdf"), "urdf", "thor.urdf.xacro")
    robot_description = ParameterValue(
        Command(["xacro ", xacro_file, " is_sim:=False use_mock:=", use_mock]), value_type=str)
    controllers = os.path.join(share, "config", "policy_controllers.yaml")

    def spawner(name):
        return Node(package="controller_manager", executable="spawner",
                    arguments=[name, "--controller-manager", "/controller_manager"])

    return LaunchDescription([
        DeclareLaunchArgument("use_mock", default_value="false",
                              description="true = sans matériel (mock_components), pour tester dans RViz"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("max_excursion_deg", default_value="0.0",
                              description="Débattement max autour de la posture de départ (0 = pas de limite)"),
        DeclareLaunchArgument("vmax_scale", default_value="1.0",
                              description="Facteur sur les vitesses max de l'URDF (< 1 pour aller plus doucement)"),
        DeclareLaunchArgument("lag_stop_deg", default_value="10.0",
                              description="Arrêt si angle lu et angle estimé s'écartent de plus de X degrés (0 = jamais)"),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[{"robot_description": robot_description}, controllers], output="screen"),
        spawner("joint_state_broadcaster"),
        spawner("arm_position_controller"),
        spawner("gripper_group_controller"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "rviz", "policy.rviz")],
             condition=IfCondition(rviz)),
        Node(package="thor_policy", executable="policy_node", output="screen",
             parameters=[{
                 "max_excursion_deg": ParameterValue(max_exc, value_type=float),
                 "vmax_scale": ParameterValue(vmax_scale, value_type=float),
                 "lag_stop_deg": ParameterValue(lag_stop, value_type=float),
             }]),
    ])
