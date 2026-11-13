"""Vérifie le cœur du nœud (policy_core) contre des données enregistrées dans Isaac Lab (record_golden.py).

Ne demande ni ROS, ni Isaac Sim, ni GPU : numpy + onnxruntime suffisent.
Usage : ros2 run thor_policy check_golden [<golden.npz> [<policy.onnx>]]
     ou python3 -m thor_policy.check_golden [<golden.npz> [<policy.onnx>]]
Sans argument : utilise les données livrées avec le paquet (data/golden_rel_model999.npz).

Contrôles
  1. observations : build_obs(q, q̇, cible, dernière action) == observation d'Isaac
  2. ONNX         : actions de l'ONNX == actions de la policy PyTorch
  3. FK           : fk_tcp(q) == position du TCP dans la simulation
  4. dynamique    : modèle du nœud (vitesse = K/D x 0,0625 x action, bornée à v_max) vs angles suivants simulés
  5. boucle fermée: la policy sur le modèle du nœud atteint-elle les cibles enregistrées ?
"""
import math
import os
import sys

import numpy as np

from thor_policy.policy_core import (
    ACTION_SCALE, CLIP_ACTIONS, DT, JOINT_NAMES, K_OVER_D, VMAX, PolicyCore, build_obs, fk_tcp)


def verdict(ok):
    return "OK " if ok else "ÉCHEC"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if len(argv) < 2:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("thor_policy")
    npz_path = argv[0] if argv else os.path.join(share, "data", "golden_rel_model999.npz")
    onnx_path = argv[1] if len(argv) > 1 else os.path.join(share, "models", "policy_rel_model999.onnx")
    g = np.load(npz_path, allow_pickle=False)
    core = PolicyCore(onnx_path)
    T, N = g["obs"].shape[:2]
    all_ok = True
    print(f"Données : {T} pas x {N} environnements, dt = {float(g['step_dt']):.4f} s | ONNX : {onnx_path}\n")

    # 0. conventions
    names = [str(n) for n in g["joint_names"]]
    ok = names == JOINT_NAMES and abs(float(g["step_dt"]) - DT) < 1e-6 and float(g["clip_actions"]) == CLIP_ACTIONS
    all_ok &= ok
    print(f"[{verdict(ok)}] conventions : articulations {names}, dt, clip_actions = {float(g['clip_actions'])}")
    quat = g["cmd"][..., 3:7].reshape(-1, 4)
    print(f"        quaternion de cible observé (min / max par composante) : {quat.min(0).round(3)} / {quat.max(0).round(3)}")

    # 1. observations
    rebuilt = np.stack([[build_obs(g["q"][t, i], g["qd"][t, i], g["cmd"][t, i, :3], g["last_action"][t, i], g["cmd"][t, i, 3:7])
                         for i in range(N)] for t in range(T)])
    d = np.abs(rebuilt - g["obs"]).max(axis=(0, 1))
    ok = d.max() < 1e-4
    all_ok &= ok
    print(f"[{verdict(ok)}] observations reconstruites : écart max {d.max():.2e} "
          f"(par bloc : q {d[:6].max():.1e}, q̇ {d[6:12].max():.1e}, cible {d[12:19].max():.1e}, action {d[19:].max():.1e})")

    la = g["last_action"][1:]; prev = np.clip(g["action"][:-1], -CLIP_ACTIONS, CLIP_ACTIONS)
    keep = ~g["done"][:-1].astype(bool)
    d = np.abs(la - prev)[keep].max()
    ok = d < 1e-5
    all_ok &= ok
    print(f"[{verdict(ok)}] « dernière action » = action précédente bornée à ±{CLIP_ACTIONS:g} : écart max {d:.2e}")

    # 2. ONNX
    a_onnx = np.stack([[core.infer(g["obs"][t, i]) for i in range(N)] for t in range(T)])
    d = np.abs(a_onnx - g["action"]).max()
    ok = d < 1e-3
    all_ok &= ok
    print(f"[{verdict(ok)}] ONNX vs policy PyTorch : écart max {d:.2e}")

    # 3. FK
    tcp = np.stack([[fk_tcp(g["q"][t, i]) for i in range(N)] for t in range(T)])
    d = 1000 * np.abs(tcp - g["tcp_b"]).max()
    ok = d < 1.0
    all_ok &= ok
    print(f"[{verdict(ok)}] FK du nœud vs TCP simulé : écart max {d:.3f} mm")

    # 4. dynamique (on exclut les pas avec reset)
    valid = ~g["done"].astype(bool)
    a = np.clip(g["action"], -CLIP_ACTIONS, CLIP_ACTIONS)
    v = np.clip(K_OVER_D * ACTION_SCALE * a, -VMAX, VMAX)
    pred = g["q"] + v * DT
    err = np.degrees(np.abs(pred - g["q_next"]))[valid]          # (échantillons, 6)
    naive = np.degrees(np.abs(g["q"] + ACTION_SCALE * a - g["q_next"]))[valid]
    real_step = np.degrees(np.abs(g["q_next"] - g["q"]))[valid]
    print("[INFO] modèle d'articulation du nœud vs simulation, par pas (degrés) :")
    print(f"        {'':10s} {'déplacement simulé méd':>22s} {'erreur modèle méd':>18s} {'p95':>7s} {'erreur « 0,0625 x a » méd':>27s}")
    for j, n in enumerate(JOINT_NAMES):
        print(f"        {n:10s} {np.median(real_step[:, j]):22.3f} {np.median(err[:, j]):18.3f} {np.percentile(err[:, j], 95):7.3f}"
              f" {np.median(naive[:, j]):27.3f}")

    # 5. boucle fermée sur le modèle, cibles enregistrées, départ bras vertical
    targets = np.unique(np.round(g["cmd"][..., :3].reshape(-1, 3), 5), axis=0)
    errs = []
    for tgt in targets:
        core.reset(np.zeros(6))
        for _ in range(240):
            q, _ = core.step(tgt)
        errs.append(100 * np.linalg.norm(fk_tcp(q) - tgt))
    errs = np.array(errs)
    ok = np.median(errs) < 1.0 and (errs > 5).mean() < 0.05
    all_ok &= ok
    print(f"[{verdict(ok)}] boucle fermée sur le modèle du nœud ({len(targets)} cibles, 8 s) : "
          f"médiane {np.median(errs):.2f} cm, p95 {np.percentile(errs, 95):.2f} cm, échecs (> 5 cm) {100 * (errs > 5).mean():.1f} %")

    print("\nRÉSULTAT :", "tout est cohérent." if all_ok else "au moins un contrôle a échoué : ne pas brancher le vrai bras.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
