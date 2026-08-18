# PIBench Scene Catalog

A living index of every scene in the benchmark, its suite, the physical concepts it tests, and how to run it.

---

## Statics Suite

### `TowerFall`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/statics/tower_fall.py` |
| **Question** | Two towers stand on a tilting platform. Which tower falls? |
| **Answer type** | Choice: `"A"`, `"B"`, `"both"`, `"neither"` |
| **Concepts** | Center of mass, support polygon, stability |
| **Textbook link** | Chapter 2 — rigid-body motions, CoM |
| **Run CLI** | `python -m pibench run --suite statics --predictor physics_oracle --n 5` |
| **Render CLI** | `python -m pibench render TowerFall --seed 0 --output output/tower_fall_seed0.png` |

**How ground truth is computed:**  
The platform tilts to a random angle. MuJoCo rolls the simulation forward until the towers settle or fall. A tower is counted as "fallen" if any of its blocks drops below the platform surface by more than a threshold.

**Latent parameters:** `tower_a_height`, `tower_b_height`, `base_a`, `base_b`, `mass_a`, `mass_b`, `tilt_angle_deg`.

---

### `SlopeSlide`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/statics/slope_slide.py` |
| **Question** | A block rests on an incline with a given angle and static-friction coefficient. Does it slide down? |
| **Answer type** | Boolean: `"yes"` / `"no"` |
| **Concepts** | Static friction, angle of repose, gravity decomposition |
| **Textbook link** | Chapter 2 — force balance on inclined plane |
| **Ground truth** | `tan(angle) > mu_s` |

**Latent parameters:** `angle_deg`, `mu_s`, `block_mass`.

---

### `SupportBalance`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/statics/support_balance.py` |
| **Question** | A uniform beam carries a point mass at an offset. Where must the support be placed so the beam balances horizontally? |
| **Answer type** | Choice: `"left of center"`, `"center"`, `"right of center"` |
| **Concepts** | Center of mass, torque balance, static equilibrium |
| **Textbook link** | Chapter 2 — rigid-body moments / static equilibrium |
| **Ground truth** | Weighted average of beam CoM and point-mass position |

**Latent parameters:** `beam_mass`, `point_mass`, `mass_offset`, `balance_x`.

---

### `ToppleDirection`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/statics/topple_direction.py` |
| **Question** | An off-center tower stands on a tilted platform. Which way does it topple? |
| **Answer type** | Choice: `"left"`, `"right"`, `"neither"` |
| **Concepts** | Center of mass, support polygon, toppling direction |
| **Textbook link** | Chapter 2 — rigid-body stability |
| **Ground truth** | Net CoM offset from overhang + platform tilt |

**Latent parameters:** `tower_height`, `overhang_direction`, `max_overhang`, `tilt_angle_deg`, `tilt_direction`.

---

## Dynamics Suite

### `PendulumSwing`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/dynamics/pendulum_swing.py` |
| **Question** | Two pendulums of different lengths and masses are released from the same angle. Which has the longer period? |
| **Answer type** | Choice: `"A"`, `"B"`, `"same"` |
| **Concepts** | Pendulum period, length scaling, mass independence |
| **Textbook link** | Chapter 3 — rotational kinematics / simple harmonic motion |
| **Ground truth** | Longer pendulum has longer period (`T ≈ 2π√(L/g)`) |

**Latent parameters:** `length_a`, `length_b`, `mass_a`, `mass_b`, `release_angle_deg`.

---

### `CollisionBounce`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/dynamics/collision_bounce.py` |
| **Question** | A moving ball collides with a stationary ball on a frictionless surface. After a perfectly elastic head-on collision, which ball moves faster? |
| **Answer type** | Choice: `"A"`, `"B"`, `"same"` |
| **Concepts** | Conservation of momentum, conservation of kinetic energy |
| **Textbook link** | Chapter 3 — momentum / impulse |
| **Ground truth** | 1D elastic collision formulas |

**Latent parameters:** `mass_a`, `mass_b`, `velocity_a`.

---

### `ProjectileHit`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/dynamics/projectile_hit.py` |
| **Question** | A ball is launched from ground level at a given speed and angle. How far does it land? |
| **Answer type** | Numeric (meters), tolerance-based scoring |
| **Concepts** | Projectile motion, range equation |
| **Textbook link** | Chapter 3 — projectile kinematics |
| **Ground truth** | `R = v² sin(2θ) / g` (validated with MuJoCo) |

**Latent parameters:** `speed`, `angle_deg`, `mass`, `analytic_range`, `simulated_range`.

---

## Contact Suite

### `PushTipVsSlide`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/contact/push_tip_vs_slide.py` |
| **Question** | A block is pushed horizontally at a given height. Does it tip over or slide? |
| **Answer type** | Choice: `"tip"`, `"slide"` |
| **Concepts** | Moment balance, line of action, friction |
| **Textbook link** | Chapter 4 — velocity kinematics / contact forces |
| **Ground truth** | MuJoCo rollout: tipped if Z-axis tilt exceeds a threshold |

**Latent parameters:** `block_mass`, `half_width`, `half_height`, `push_height`, `max_tilt_deg`.

---

### `StackStability`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/contact/stack_stability.py` |
| **Question** | A stack of blocks is tapped from the side by a moving ball. Does it remain standing? |
| **Answer type** | Boolean: `"yes"` / `"no"` |
| **Concepts** | Impulse, support polygon, stacking stability |
| **Textbook link** | Chapter 4 — contact / impulse |
| **Ground truth** | MuJoCo rollout: collapsed if any block displacement exceeds a threshold |

**Latent parameters:** `n_blocks`, `block_mass`, `ball_mass`, `ball_speed`, `impact_height`, `max_displacement`.

---

### `WedgeInsert`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/contact/wedge_insert.py` |
| **Question** | A triangular wedge is pushed into a gap. Does it fit through or jam? |
| **Answer type** | Choice: `"fits"`, `"jams"` |
| **Concepts** | Contact geometry, clearance, jamming |
| **Textbook link** | Chapter 4 — contact constraints |
| **Ground truth** | MuJoCo rollout: fits if wedge tip advances past the gap |

**Latent parameters:** `wedge_base_width`, `wedge_length`, `gap_width`, `wedge_delta_x`.

---

### `FrictionPile`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/contact/friction_pile.py` |
| **Question** | Three objects with different mass and friction are pushed. Which is hardest to start moving? |
| **Answer type** | Choice: `"A"`, `"B"`, `"C"` |
| **Concepts** | Static friction, normal force, Coulomb friction |
| **Textbook link** | Chapter 4 — contact / friction |
| **Ground truth** | Object maximizing `mu_s * mass` |

**Latent parameters:** `masses`, `mu_s`, `thresholds`.

---

### `SlipGrip`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/contact/slip_grip.py` |
| **Question** | A gripper squeezes a block and lifts. Does the block lift or slip? |
| **Answer type** | Choice: `"lift"`, `"slip"` |
| **Concepts** | Coulomb friction, grip force, weight |
| **Textbook link** | Chapter 4 — contact / friction |
| **Ground truth** | Analytic: lift if `2 * mu * F_grip >= m * g` |

**Latent parameters:** `block_mass`, `grip_force`, `mu_s`, `weight`, `total_friction_capacity`.

---

## Articulated Suite

### `DrawerPull`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/articulated/drawer_pull.py` |
| **Question** | A drawer is pulled with a constant force. Does it open or jam? |
| **Answer type** | Boolean: `"yes"` / `"no"` |
| **Concepts** | Prismatic joints, static friction, actuators |
| **Textbook link** | Chapter 5 — constraints / inverse kinematics |
| **Ground truth** | MuJoCo rollout: opens if displacement exceeds threshold |

**Latent parameters:** `drawer_mass`, `applied_force`, `frictionloss`, `max_displacement`.

---

### `DoorSwing`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/articulated/door_swing.py` |
| **Question** | A door is pushed with a constant torque. Does it swing open or stick? |
| **Answer type** | Boolean: `"yes"` / `"no"` |
| **Concepts** | Revolute joints, hinge friction, torque, angular motion |
| **Textbook link** | Chapter 5 — constraints / inverse kinematics |
| **Ground truth** | MuJoCo rollout: opens if angular displacement exceeds threshold |

**Latent parameters:** `door_mass`, `applied_torque`, `frictionloss`, `max_angle_deg`.

---

### `RopeTension`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/articulated/rope_tension.py` |
| **Question** | Two masses hang from a rope over a pulley. Which mass descends? |
| **Answer type** | Choice: `"A"`, `"B"`, `"same"` |
| **Concepts** | Tension, pulleys, constrained motion, gravity |
| **Textbook link** | Chapter 5 — constraints / tendons |
| **Ground truth** | MuJoCo rollout: compares final average heights of the two masses |

**Latent parameters:** `mass_a`, `mass_b`, `final_z_a`, `final_z_b`.

---

### `GearTurn`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/articulated/gear_turn.py` |
| **Question** | Gear A is driven counter-clockwise and meshes externally with gear B. Which way does gear B turn? |
| **Answer type** | Choice: `"clockwise"`, `"counter-clockwise"` |
| **Concepts** | External gear meshing, rotation direction, kinematic constraints |
| **Textbook link** | Chapter 5 — constraints / kinematic chains |
| **Ground truth** | External meshing produces opposite rotation: gear B turns clockwise |

**Latent parameters:** `radius_a`, `radius_b`, `mass_a`, `mass_b`, `applied_torque`, `gear_a_rotation_deg`.

---

## Deformable Suite

### `ChainDrape`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/deformable/chain_drape.py` |
| **Question** | A chain is draped over a horizontal bar. How high is the free end above the floor? |
| **Answer type** | Numeric (meters), tolerance-based scoring |
| **Concepts** | Deformable-body approximation, contact, sag geometry |
| **Textbook link** | Chapter 5 — introduction to deformables / constraints |
| **Ground truth** | MuJoCo rollout: final Z height of the chain's free end |

**Latent parameters:** `n_segments`, `segment_mass`, `segment_radius`, `segment_half_length`, `bar_height`, `free_end_height`.

---

## Parameter Estimation / Counterfactual Suite

*Coming in Phase 5.*

---

## Concept coverage map

| Concept | TowerFall | SlopeSlide | SupportBalance | ToppleDirection | CollisionBounce | PendulumSwing | ProjectileHit | PushTipVsSlide | StackStability | WedgeInsert | FrictionPile | SlipGrip | DrawerPull | DoorSwing | RopeTension | GearTurn | ChainDrape |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Center of mass | ✅ | ✅ | ✅ | ✅ | | | | ✅ | ✅ | | | | | | | | |
| Support polygon | ✅ | | ✅ | ✅ | | | | | ✅ | | | | | | | | |
| Friction / angle of repose | | ✅ | | | | | | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | | | |
| Torque balance | | | ✅ | | | | | ✅ | | | | | | ✅ | | | |
| Toppling direction | | | | ✅ | | | | ✅ | | | | | | | | | |
| Conservation of momentum | | | | | ✅ | | | | ✅ | | | | | | | | |
| Conservation of energy | | | | | ✅ | ✅ | | | | | | | | | | | |
| Period / length scaling | | | | | | ✅ | | | | | | | | | | | |
| Projectile range | | | | | | | ✅ | | | | | | | | | | |
| Contact geometry / jamming | | | | | | | | | | ✅ | | | | | | | |
| Grip friction / static threshold | | | | | | | | | | | ✅ | ✅ | | | | | |
| Prismatic / hinge constraints | | | | | | | | | | | | | ✅ | ✅ | | | |
| Tendons / pulleys | | | | | | | | | | | | | | | ✅ | | |
| Gear meshing / kinematic chains | | | | | | | | | | | | | | | | ✅ | |
| Deformable-body approximation | | | | | | | | | | | | | | | | | ✅ |

---

*Last updated: 2026-08-18*
