"""Cœur de la policy de reach de Thor, sans ROS : observations, inférence ONNX, modèle d'articulation, FK.

Tout ici reproduit la tâche Isaac Lab Template-Reach-Thor-Rel-v0 (run 2026-11-06_21-54-28_rel, model_999) :
- observation (25) = [q - q_défaut (6), q̇ (6), cible xyz (3) + quaternion wxyz (4) dans base_link, dernière action (6)]
- action relative : consigne = q + 0,0625 x action, action bornée à ±6 (clip_actions du wrapper rsl_rl)
- IMPORTANT : en simulation, la consigne relative est réappliquée à chaque sous-pas physique à partir de
  l'angle MESURÉ, et le correcteur PD (raideur K = 400, amortissement D = 40) la transforme en VITESSE :
  v ≈ (K / D) x 0,0625 x action = 0,625 x action (rad/s), puis PhysX borne v aux vitesses de l'URDF.
  Vérifié le 13 nov. 2026 en boucle fermée sur ce modèle : 0,38 cm médiane, 0 % d'échecs (Isaac : 0,38 cm),
  alors qu'un déplacement de 0,0625 x action par pas donne 8 cm et une policy qui oscille.
- En boucle ouverte, l'angle « actuel » est l'angle estimé = dernière consigne envoyée.
"""
import math

import numpy as np

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
DEFAULT_Q = np.zeros(6)                                  # posture par défaut de l'entraînement (bras vertical)
LIMITS = np.array([2.967, math.pi / 2, math.pi / 2, 2.967, math.pi / 2, 2.967])   # butées URDF (rad)
VMAX = np.array([0.58, 0.23, 0.23, 0.58, 1.45, 1.45])   # vitesses max URDF (rad/s)
ACTION_SCALE = 0.0625
K_OVER_D = 400.0 / 40.0                                  # raideur / amortissement de l'actionneur simulé
CLIP_ACTIONS = 6.0
DT = 1.0 / 30.0
TARGET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])            # orientation de cible de l'entraînement (non utilisée)

# Zone de cibles vue à l'entraînement (CylindricalPoseCommandCfg), repère base_link
R_RANGE = (0.25, 0.40)
AZ_RANGE_DEG = (45.0, 135.0)                             # azimut depuis +x
Z_RANGE = (0.05, 0.40)

# Chaîne cinématique de l'URDF : origines = décalages purs en z, axes z,x,x,z,x,z
_OFFSETS_Z = [0.099, 0.103, 0.16, 0.0895, 0.1045, 0.0135]
_AXES = "zxxzxz"
_TCP_Z = 0.043 + 0.075                                   # joint_6 -> gripper_base -> gripper_mid_point


def _rot(axis, t):
    c, s = math.cos(t), math.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk_tcp(q):
    """Position de gripper_mid_point dans base_link (m)."""
    R, p = np.eye(3), np.zeros(3)
    for i in range(6):
        p = p + R @ np.array([0.0, 0.0, _OFFSETS_Z[i]])
        R = R @ _rot(_AXES[i], q[i])
    return p + R @ np.array([0.0, 0.0, _TCP_Z])


def target_in_zone(xyz, margin=0.0):
    """True si la cible est dans la zone vue à l'entraînement (la policy n'a rien appris ailleurs)."""
    x, y, z = xyz
    r = math.hypot(x, y)
    az = math.degrees(math.atan2(y, x))
    return (R_RANGE[0] - margin <= r <= R_RANGE[1] + margin
            and AZ_RANGE_DEG[0] - math.degrees(margin / max(r, 1e-6)) <= az <= AZ_RANGE_DEG[1] + math.degrees(margin / max(r, 1e-6))
            and Z_RANGE[0] - margin <= z <= Z_RANGE[1] + margin)


def build_obs(q, qd, target_xyz, last_action, target_quat=TARGET_QUAT):
    return np.concatenate([np.asarray(q) - DEFAULT_Q, qd, target_xyz, target_quat, last_action]).astype(np.float32)


class PolicyCore:
    """Policy + modèle d'articulation en boucle ouverte. Un appel à step() = un pas de 1/30 s."""

    def __init__(self, onnx_path, safety_factor=0.95, vmax_scale=1.0):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.lo, self.hi = -safety_factor * LIMITS, safety_factor * LIMITS
        self.vmax = vmax_scale * VMAX
        self.reset(np.zeros(6))

    def infer(self, obs):
        return self.sess.run(None, {self.in_name: obs[None, :].astype(np.float32)})[0][0]

    def reset(self, q0, excursion=None):
        """Repart de q0 (angles estimés). excursion (rad, par articulation) : débattement max autour de q0."""
        self.q = np.array(q0, dtype=float)
        self.qd = np.zeros(6)
        self.last_a = np.zeros(6)
        if excursion is None:
            self.box_lo, self.box_hi = self.lo, self.hi
        else:
            self.box_lo = np.maximum(self.lo, self.q - excursion)
            self.box_hi = np.minimum(self.hi, self.q + excursion)

    def step(self, target_xyz):
        obs = build_obs(self.q, self.qd, target_xyz, self.last_a)
        a_raw = self.infer(obs)
        a = np.clip(a_raw, -CLIP_ACTIONS, CLIP_ACTIONS)
        v = np.clip(K_OVER_D * ACTION_SCALE * a, -self.vmax, self.vmax)   # vitesse demandée (rad/s)
        q_cmd = np.clip(self.q + v * DT, self.box_lo, self.box_hi)
        self.qd = (q_cmd - self.q) / DT
        self.q = q_cmd
        self.last_a = a
        return q_cmd.copy(), a
