"""Nœud ROS 2 qui exécute la policy de reach de Thor (ONNX, 30 Hz, boucle ouverte).

Entrées
  /joint_states                    (sensor_msgs/JointState)   angles lus (RRF via thor_interface, ou mock)
  /thor_policy/target              (geometry_msgs/PointStamped) cible du TCP dans base_link (m)
  /thor_policy/enable              (std_srvs/SetBool)          démarre / arrête la policy
Sorties
  <command_topic>                  (std_msgs/Float64MultiArray) consignes joint_1..joint_6 (rad)
  /thor_policy/markers             (visualization_msgs/MarkerArray) cible (vert) et TCP estimé (bleu)
  /thor_policy/error_cm            (std_msgs/Float64)  distance TCP estimé - cible
  /thor_policy/lag_deg             (std_msgs/Float64)  écart max entre angle lu et angle estimé

Sécurité
  - la policy ne démarre qu'avec /thor_policy/enable, une cible valide et des /joint_states reçus ;
  - cible refusée hors de la zone d'entraînement (r 0,25-0,40 m, azimut 45-135°, z 0,05-0,40 m) ;
  - consignes bornées à safety_factor x butées URDF, et à ±max_excursion_deg autour de la posture de départ ;
  - vitesse bornée aux vitesses URDF (x vmax_scale) ;
  - arrêt automatique si l'angle lu s'écarte de l'angle estimé de plus de lag_stop_deg (file RRF qui
    s'allonge, articulation bloquée...). Mettre lag_stop_deg <= 0 pour désactiver (mock).
"""
import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray

from thor_policy.policy_core import DT, JOINT_NAMES, PolicyCore, fk_tcp, target_in_zone


class ThorPolicyNode(Node):
    def __init__(self):
        super().__init__("thor_policy")
        default_model = os.path.join(get_package_share_directory("thor_policy"), "models", "policy_rel_model999.onnx")
        self.declare_parameter("model_path", default_model)
        self.declare_parameter("command_topic", "/arm_position_controller/commands")
        self.declare_parameter("safety_factor", 0.95)
        self.declare_parameter("vmax_scale", 1.0)
        self.declare_parameter("max_excursion_deg", 0.0)   # 0 = pas de limite autour de la posture de départ
        self.declare_parameter("lag_stop_deg", 10.0)
        self.declare_parameter("log_period_s", 1.0)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.core = PolicyCore(p("model_path"), safety_factor=p("safety_factor"), vmax_scale=p("vmax_scale"))
        self.get_logger().info(f"Policy chargée : {p('model_path')}")

        self.q_meas = None
        self.target = None
        self.enabled = False
        self.steps = 0

        self.cmd_pub = self.create_publisher(Float64MultiArray, p("command_topic"), 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/thor_policy/markers", 10)
        self.err_pub = self.create_publisher(Float64, "/thor_policy/error_cm", 10)
        self.lag_pub = self.create_publisher(Float64, "/thor_policy/lag_deg", 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)
        self.create_subscription(PointStamped, "/thor_policy/target", self.on_target, 10)
        self.create_service(SetBool, "/thor_policy/enable", self.on_enable)
        self.timer = self.create_timer(DT, self.on_timer)
        self.get_logger().info("Prêt. 1) publier une cible sur /thor_policy/target  2) appeler /thor_policy/enable true")

    # ------------------------------------------------------------------ entrées
    def on_joint_states(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(n in idx for n in JOINT_NAMES):
            self.q_meas = np.array([msg.position[idx[n]] for n in JOINT_NAMES])

    def on_target(self, msg):
        if msg.header.frame_id not in ("", "base_link"):
            self.get_logger().warn(f"Cible refusée : repère '{msg.header.frame_id}', attendu 'base_link'.")
            return
        xyz = np.array([msg.point.x, msg.point.y, msg.point.z])
        if not target_in_zone(xyz):
            r, az = math.hypot(xyz[0], xyz[1]), math.degrees(math.atan2(xyz[1], xyz[0]))
            self.get_logger().warn(
                f"Cible refusée (r {r:.3f} m, azimut {az:.0f}°, z {xyz[2]:.3f} m) : hors de la zone d'entraînement "
                "r 0,25-0,40 m, azimut 45-135°, z 0,05-0,40 m.")
            return
        self.target = xyz
        self.get_logger().info(f"Nouvelle cible : {np.round(xyz, 3).tolist()} m")

    def on_enable(self, req, resp):
        if req.data:
            if self.q_meas is None:
                resp.success, resp.message = False, "Aucun /joint_states reçu."
            elif self.target is None:
                resp.success, resp.message = False, "Aucune cible valide reçue sur /thor_policy/target."
            else:
                exc = self.get_parameter("max_excursion_deg").value
                self.core.reset(self.q_meas, excursion=math.radians(exc) if exc > 0 else None)
                self.enabled, self.steps = True, 0
                resp.success = True
                resp.message = f"Policy démarrée depuis {np.round(np.degrees(self.q_meas), 1).tolist()} deg."
        else:
            self.enabled = False
            resp.success, resp.message = True, "Policy arrêtée (le contrôleur garde la dernière consigne)."
        self.get_logger().info(resp.message)
        return resp

    # ------------------------------------------------------------------ boucle 30 Hz
    def on_timer(self):
        if not self.enabled:
            return
        q_cmd, _ = self.core.step(self.target)
        self.cmd_pub.publish(Float64MultiArray(data=q_cmd.tolist()))
        self.steps += 1

        tcp = fk_tcp(q_cmd)
        err_cm = 100.0 * float(np.linalg.norm(tcp - self.target))
        self.err_pub.publish(Float64(data=err_cm))
        lag = float(np.degrees(np.max(np.abs(self.q_meas - q_cmd)))) if self.q_meas is not None else float("nan")
        self.lag_pub.publish(Float64(data=lag))
        self.publish_markers(tcp)

        lag_stop = self.get_parameter("lag_stop_deg").value
        if lag_stop > 0 and lag > lag_stop and self.steps > 15:
            self.enabled = False
            self.get_logger().error(
                f"ARRÊT : l'angle lu s'écarte de {lag:.1f}° de l'angle estimé (seuil {lag_stop}°). "
                "File de mouvements RRF trop longue, articulation bloquée, ou pas perdus ?")
            return
        period = max(1, int(round(self.get_parameter("log_period_s").value / DT)))
        if self.steps % period == 0:
            self.get_logger().info(f"t={self.steps * DT:5.1f} s | erreur estimée {err_cm:5.2f} cm | écart lu/estimé {lag:5.2f}°")

    def publish_markers(self, tcp):
        arr = MarkerArray()
        for i, (pos, rgb, size) in enumerate([(self.target, (0.1, 0.9, 0.1), 0.02), (tcp, (0.1, 0.4, 1.0), 0.015)]):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = "thor_policy", i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = size
            m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, 0.9)
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = ThorPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
