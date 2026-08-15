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

*Coming in Phase 3.*

## Articulated / Deformable Suite

*Coming in Phase 4.*

## Parameter Estimation / Counterfactual Suite

*Coming in Phase 5.*

---

## Concept coverage map

| Concept | TowerFall | SlopeSlide | SupportBalance | ToppleDirection | CollisionBounce | PendulumSwing | ProjectileHit |
|---|---|---|---|---|---|---|---|
| Center of mass | ✅ | ✅ | ✅ | ✅ | | | |
| Support polygon | ✅ | | ✅ | ✅ | | | |
| Friction / angle of repose | | ✅ | | | | | |
| Torque balance | | | ✅ | | | | |
| Toppling direction | | | | ✅ | | | |
| Conservation of momentum | | | | | ✅ | | |
| Conservation of energy | | | | | ✅ | ✅ | |
| Period / length scaling | | | | | | ✅ | |
| Projectile range | | | | | | | ✅ |

---

*Last updated: 2026-08-13*
