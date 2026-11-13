# thor_policy : déployer la policy de reach sur Thor

Policy : run Isaac Lab `2026-11-06_21-54-28_rel`, `model_999` (action relative), exportée en `models/policy_rel_model999.onnx`.
Le nœud tourne à 30 Hz sur CPU (pas de GPU, pas d'Isaac Sim). Il reste dans l'espace articulaire ROS
(joint_1…joint_6, rad) : c'est `thor_interface` qui traduit vers RRF.

## Le point clé : comment le nœud imite la simulation

En simulation, l'action relative est réappliquée à chaque sous-pas à partir de l'angle **mesuré**, et le
correcteur PD (raideur 400, amortissement 40) la transforme en **vitesse** :

    v = (400 / 40) x 0,0625 x action = 0,625 x action   (rad/s), bornée aux vitesses de l'URDF

Le nœud fait donc, à chaque pas de 1/30 s : `consigne = angle estimé + v x 1/30`, puis borne la consigne
à 95 % des butées. En boucle ouverte, l'angle estimé est la dernière consigne envoyée.
Vérifié en boucle fermée sur ce modèle : 0,38 cm de médiane, 0 % d'échecs (comme dans Isaac). Avec
« consigne = angle + 0,0625 x action », la policy oscille et rate de 8 cm : c'est cette erreur qu'on évite.
Chaque segment envoyé à RRF ne dépasse jamais ce que l'articulation peut faire en 1/30 s : la file de
mouvements de RRF ne doit pas s'allonger (si elle s'allonge, `lag_deg` monte et le nœud s'arrête).

---

## 1. À la maison (TomLinux)

### 1a. Données de référence (terminal Isaac, `conda activate env_isaaclab`)
```bash
cd ~/workspace/thor_rl
python scripts/rsl_rl/record_golden.py --task Template-Reach-Thor-Rel-Play-v0 --headless --num_envs 8 \
  --checkpoint ~/workspace/thor_rl/logs/rsl_rl/thor_reach/2026-11-06_21-54-28_rel/model_999.pt
```
Produit `~/workspace/thor_rl/logs/golden/golden_rel_model999.npz` (900 pas x 8 environnements).

### 1b. Installation et compilation (terminal ROS, SANS conda : `conda deactivate` jusqu'à disparition du préfixe)
```bash
source /opt/ros/humble/setup.bash
python3 -m pip install --user "numpy>=1.21.6,<2" "onnxruntime==1.19.2"
sudo apt install ros-humble-ros2-controllers ros-humble-ros2-control   # si position_controllers manque
cd ~/Thor-ROS/ws_thor
colcon build --packages-select thor_urdf thor_policy
source install/setup.bash
```

### 1c. Vérification contre la simulation (aucun matériel)
```bash
ros2 run thor_policy check_golden
```
Sans argument, il utilise les données de référence livrées avec le paquet (`data/golden_rel_model999.npz`,
enregistrées le 13 nov. 2026). Pour d'autres données : `ros2 run thor_policy check_golden <fichier.npz>`.
Attendu : toutes les lignes `[OK ]` et « tout est cohérent ». La ligne « modèle d'articulation » montre
l'erreur du modèle du nœud par rapport à la simulation, pas par pas.

### 1d. Test de toute la chaîne ROS, bras simulé (mock_components)
```bash
ros2 launch thor_policy policy_robot.launch.py use_mock:=true lag_stop_deg:=0
# autre terminal (ROS, sourcé)
ros2 topic pub --once /thor_policy/target geometry_msgs/msg/PointStamped "{header: {frame_id: base_link}, point: {x: 0.0, y: 0.32, z: 0.22}}"
ros2 service call /thor_policy/enable std_srvs/srv/SetBool "{data: true}"
```
Attendu : dans RViz, le bras se plie vers la sphère verte ; les logs affichent une erreur estimée qui
descend sous 1 cm en 4 à 5 s. Changer de cible en republiant sur `/thor_policy/target` (le nœud continue).
Arrêt : `ros2 service call /thor_policy/enable std_srvs/srv/SetBool "{data: false}"`.

---

## 2. Transfert vers le Dell (par git)

Tout est sur le fork `markestom0512-max/Thor-ROS`, branche **`policy-deploy`** : le paquet `thor_policy`
(policy ONNX et données de référence comprises) et les ajouts `use_mock` dans le xacro. La branche part de
`main` du fork, qui contient déjà la correction « angles en flottant » de `thor_interface.cpp`.

Sur le Dell, on ne récupère **que le dossier du paquet**, pour ne pas toucher aux modifications locales
(port série, board_type) :
```bash
cd ~/Thor-ROS              # chemin du dépôt sur le Dell, à adapter
git remote -v              # repérer le nom du remote qui pointe vers markestom0512-max/Thor-ROS
                           # s'il n'existe pas : git remote add fork https://github.com/markestom0512-max/Thor-ROS.git
git fetch fork policy-deploy
git checkout fork/policy-deploy -- ws_thor/src/thor_policy
```
(remplacer `fork` par le nom trouvé avec `git remote -v`). Puis, pour le xacro, **ne pas écraser** les
fichiers du Dell : reporter à la main les deux ajouts suivants (`git diff fork/main fork/policy-deploy --
ws_thor/src/thor_urdf` les montre).

`thor_urdf/urdf/thor.urdf.xacro`, sous la ligne `<xacro:arg name="is_sim" .../>` :
```xml
<xacro:arg name="use_mock" default="false"/>
```
`thor_urdf/urdf/thor_arm.ros2_control.xacro`, entourer le bloc `<hardware>` ThorInterface existant :
```xml
<xacro:unless value="$(arg is_sim)">
  <xacro:if value="$(arg use_mock)">
    <hardware><plugin>mock_components/GenericSystem</plugin></hardware>
  </xacro:if>
  <xacro:unless value="$(arg use_mock)">
    <!-- le bloc <hardware> ThorInterface du Dell, inchangé -->
  </xacro:unless>
</xacro:unless>
```
Puis les mêmes commandes qu'en 1b (pip, onnxruntime, colcon build), et `ros2 run thor_policy check_golden` :
il doit donner exactement les mêmes chiffres que sur TomLinux.

---

## 3. Au kot, avant la policy (dans cet ordre)

1. **Relever la config RRF** (console RRF) : `M92`, `M203`, `M201`, `M566`, `M208`, `M584`, `M569 P0.0` … `P0.6`.
   `M203` doit être **au moins** égal aux vitesses de l'URDF (en °/min si les axes sont en degrés) :
   j1 1992, j2 791, j3 791, j4 1992, j5 et j6 4985 (attention au différentiel de j5/j6).
2. **Résolution des consignes** : `grep -n "static_cast<int>" ws_thor/src/thor_controller/src/thor_interface.cpp`.
   Le commit 744d7ef du fork (7 oct.) envoie des angles en flottant ; si le Dell en est resté à la version
   d'AngelLM, `write()` tronque au degré entier, soit environ 7 mm d'erreur au bout du bras.
3. **Que renvoie M408 ?** Envoyer un `G0` lent sur un axe et interroger `M408` pendant le mouvement :
   la position change-t-elle en cours de route, ou seulement à la fin ?
4. **Zéro et sens de chaque articulation, à petite amplitude**, policy à l'arrêt :
   ```bash
   ros2 launch thor_policy policy_robot.launch.py
   ros2 topic pub --once /arm_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]}"
   ```
   Une articulation à la fois (0,1 rad ≈ 6°), revenir à 0 entre chaque. Le vrai bras et RViz doivent bouger
   **dans le même sens**. Vérifier aussi que « tout à 0 » = bras vertical.
5. **Premier essai de la policy, bridé** : la main près de l'alimentation 24 V.
   ```bash
   ros2 launch thor_policy policy_robot.launch.py vmax_scale:=0.5 max_excursion_deg:=15
   ```
   À mi-vitesse, la policy reste stable et aussi précise, mais met environ 2 fois plus de temps (vérifié sur le
   modèle : 0,42 cm de médiane en 16 s, contre 8 s à pleine vitesse).
   Cible proche, côté +y de la base (le côté où la policy a été entraînée) : vérifier que l'espace est libre.
   Puis augmenter `max_excursion_deg` (30, puis 0 = sans limite) et `vmax_scale` (1.0).

## Arrêt d'urgence
- Couper le **24 V** (le plus sûr).
- `ros2 service call /thor_policy/enable std_srvs/srv/SetBool "{data: false}"` arrête la policy ; le bras garde la dernière consigne.
- Éviter `M18`/`M84` : les moteurs ne tiennent plus, l'épaule et le coude tombent.

## Lire les logs du nœud
- `erreur estimée` : distance entre la cible et le TCP **calculé** à partir des consignes (la vraie erreur
  peut être plus grande : jeu, pas perdus ; à mesurer sur la table).
- `écart lu/estimé` : écart max entre l'angle renvoyé par RRF (`M408`) et l'angle estimé. S'il grandit,
  la file de mouvements RRF prend du retard ; au-delà de `lag_stop_deg` (10° par défaut), le nœud s'arrête.

## Paramètres du nœud
| Paramètre | Défaut | Rôle |
|---|---|---|
| `model_path` | `share/thor_policy/models/policy_rel_model999.onnx` | policy ONNX |
| `command_topic` | `/arm_position_controller/commands` | consignes (Float64MultiArray, 6 valeurs) |
| `safety_factor` | 0,95 | consignes bornées à 95 % des butées URDF (0,90 dégrade la policy : 12 % d'échecs) |
| `vmax_scale` | 1,0 | facteur sur les vitesses URDF |
| `max_excursion_deg` | 0 | débattement max autour de la posture de départ (0 = sans limite) |
| `lag_stop_deg` | 10 | arrêt si angle lu et angle estimé s'écartent de plus de X° (0 = jamais) |
