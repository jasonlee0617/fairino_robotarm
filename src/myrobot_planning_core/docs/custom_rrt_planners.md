# Fairino custom RRT planners

This note documents the Fairino custom planners exposed through the `fairino`
MoveIt planning pipeline.

## Planner ids

Canonical planner ids:

- `mire_biait*`
- `aapf_birrt*`
- `birrt*`
- `rrt`
- `rrt*`
- `informed_rrt*`
- `prm`

Compatibility aliases:

- `mire_biait*` -> `mire_biait*`
- `aapf_birrt*` -> `aapf_birrt*`
- `birrt*` -> `birrt*`
- `rrt` -> `rrt`
- `rrt*` -> `rrt*`
- `informed-rrt*` -> `informed_rrt*`
- `prm` -> `prm`

An empty `planner_id` also selects `birrt*`.

The C++ classes are `MireBiAitStar`, `AapfBiRRTStar`, `BiRRTStar`, `RRT`,
`RRTStar`, `InformedRRTStar`, and `PRM`. `rrt` is
standard single-tree RRT and never rewires; `informed_rrt*` reuses RRT* before
the first solution and samples the multi-root informed set afterwards; `prm`
builds a fresh single-query roadmap for each request.

## Parameter files

Algorithm-specific parameters live in:

- `myrobot_planning_core/config/birrt*_params.yaml`
- `myrobot_planning_core/config/rrt_params.yaml`
- `myrobot_planning_core/config/rrt*_params.yaml`
- `myrobot_planning_core/config/prm_params.yaml`
- `myrobot_planning_core/config/mire_biait*_params.yaml`
- `myrobot_planning_core/config/aapf_birrt*_params.yaml`

The file names intentionally keep `*`. In the static Gazebo demo launch, set
`NODE_PARAMS["default_planner_id"]` to `mire_biait*`, `aapf_birrt*`,
`birrt*`, `rrt`, `rrt*`, `informed_rrt*`, `prm`.

ROS parameter namespaces avoid `*` and use stable internal keys:

- `fairino.algorithms.mire_biait_star.*`
- `fairino.algorithms.birrt_star.*`
- `fairino.algorithms.rrt.*`
- `fairino.algorithms.rrt_star.*`
- `fairino.algorithms.prm.*`

`common_planning_params.yaml` contains shared optimizer, trajectory, and pipeline settings. The launch
stack should load it first, then load `mire_biait*_params.yaml`, `aapf_birrt*_params.yaml`,
`birrt*_params.yaml`, `rrt_params.yaml`,
`rrt*_params.yaml`, and `prm_params.yaml`.

AAPF-BiRRT* uses only two sampling sources. `Guided` follows the selected IK
root while a padded MoveIt robot-world clearance query supplies a full-link
joint-space escape gradient. `Global` uses uniform joint-space samples with a
small opposite-tree bias. The two trees schedule these sources independently;
Tube/Mixed sampling and intermediate pose IK are not part of AAPF.
With `aapf.enable=true`, the collision backend must provide padded robot-world
clearance gradients; unsupported backends fail explicitly. Set it to `false`
only for the global-only ablation.

## Multiple static obstacles

The official obstacle input path is the MoveIt `PlanningScene`.

`rrt`, `rrt*`, `informed_rrt*`, `prm`, and `birrt*` consume all valid static collision objects in the
scene. Non-box shapes and boxes smaller than
`planner.min_obstacle_size_threshold` are filtered and counted in the logs.

The planning demo accepts the same compact obstacle text through the selected
YAML scene configuration used by `motion_planning_demo_sim.launch.py`.

Format:

```text
name:x,y,z:sx,sy,sz;name2:x,y,z:sx,sy,sz
```

The older single-obstacle arguments remain supported:

- `obstacle_name`
- `obstacle_position`
- `obstacle_size`

## Verification

Check parameter injection:

```bash
ros2 param list /move_group_fairino/move_group | grep fairino.algorithms
ros2 param get /move_group_fairino/move_group fairino.algorithms.birrt_star.max_iterations
ros2 param get /move_group_fairino/move_group fairino.algorithms.rrt.max_iterations
ros2 param get /move_group_fairino/move_group fairino.algorithms.rrt_star.max_iterations
ros2 param get /move_group_fairino/move_group fairino.algorithms.prm.k_neighbors
```

Expected logs:

- `selected_planner=birrt*`, `selected_planner=rrt`, `selected_planner=rrt*`, `selected_planner=informed_rrt*`, or `selected_planner=prm`
- `Planning obstacles aggregated: obs_count=...`
- `Planner branch selected: birrt* multi` or `Planner branch selected: rrt* multi`

The simulation benchmark intentionally excludes standard `rrt`; its comparison
set is `rrt*`, `informed_rrt*`, `birrt*`, `aapf_birrt*`, `mire_biait*`, and
`prm`. Every benchmark planner receives the same absolute 15 s request deadline; PRM timing includes rebuilding its roadmap for each request. The 50,000-work-unit limit is an abnormal-run safety guard, not the primary stopping rule.

There is no fixed post-solution attempt budget. After the first valid solution, planners retain the incumbent and continue improving until the shared deadline. MIRE records its first and final raw costs,
post-solution sampling, lower-bound certificate, and collision
checks in diagnostics. Its multi-root heuristic is an application extension of
existing informed and bidirectional search methods; this documentation does not
claim novelty or universal superiority.

Planning failures remain explicit. Do not add hidden global fallback behavior to
mask failed IK, collision, or sampling conditions.
