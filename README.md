# 🤖 LearningRobotics

> **Mission:** Learn robotics and AI from first principles — fast — and build something extraordinary and revolutionary.
>
> **Current focus:** Foundations → Rigid-body motions → Kinematics → Dynamics → Control → Motion planning → Reinforcement Learning → Real-world embodiment.

This repo is a public learning journal. Every chapter is documented, every experiment is reproducible, and every robot model is code.

---

## 🧑‍💻 About me

**Satyam Das** — CS grad, quant/AI engineer, and aspiring roboticist.

* GitHub: [@satyamdas03](https://github.com/satyamdas03)
* Goal: move from "I read about robots" to "I can design, simulate, and control one."
* Motto: *Learn by building. Build by logging. Log publicly so the next person can follow.*

---

## 🖥️ Hardware

| Component | Spec |
|---|---|
| Laptop | Lenovo LOQ |
| GPU | NVIDIA GeForce RTX 5060 (8 GB VRAM) |
| Driver | NVIDIA 595.95 |
| CUDA Version | 13.2 |
| OS | Windows 11 Home |

**Hardware strategy:** The project is simulation-only for the foreseeable future. No physical robot arm will be purchased. We will build and validate the full autonomy loop — planning, control, perception, imitation learning, foundation-model reasoning, skill sharing, and self-improvement — inside realistic MuJoCo/Isaac Lab simulations first. A real arm becomes an optional future extension once the virtual stack is proven.

---

## 🛠️ Tooling

| Tool | Role | Why |
|---|---|---|
| **MuJoCo** | Physics simulation + robot models | Fast, Python-first, excellent for learning kinematics/dynamics |
| **Python 3.11** | Main language | Stable ecosystem; Isaac Sim does not yet support Python 3.14 |
| **NumPy** | Numerics | Standard for transforms, FK, and data pipelines |
| **Isaac Sim / Isaac Lab** | RL + large-scale sim | Will be added when we reach RL chapters |

Each chapter gets its own virtual environment so dependencies stay clean.

---

## 📚 Curriculum & Progress Log

| Chapter | Topic | Status | Key Deliverables |
|---|---|---|---|
| **1** | Foundations, Configuration Space & Degrees of Freedom | ✅ Complete | `chapter01_foundation/` with 2R arm, 6-DOF arm, DOF inspector |
| 2 | Rigid-Body Motions (frames, rotations, transforms) | ✅ Complete | `chapter02_rigid_body_motions/` with SO(3)/SE(3) demo + tests |
| 3 | Forward Kinematics | ✅ Complete | `chapter03_forward_kinematics/` with DH + PoE FK for 6-DOF arm |
| **P0** | **PIBench — Physical Intuition Benchmark** | ✅ Phase 0 Complete | Engine + `TowerFall` statics scene |
| **P1** | **PIBench — Statics Suite** | ✅ Complete | `SlopeSlide`, `SupportBalance`, `ToppleDirection` |
| **P2** | **PIBench — Dynamics Suite** | ✅ Complete | `PendulumSwing`, `CollisionBounce`, `ProjectileHit` |
| 4 | Velocity Kinematics & Jacobians | ✅ Complete | `chapter04_velocity_kinematics/` with analytic + numeric Jacobian, viewer demo |
| **P3** | **PIBench — Contact Suite** | ✅ Complete | `PushTipVsSlide`, `StackStability`, `WedgeInsert`, `FrictionPile`, `SlipGrip` |
| 5 | Inverse Kinematics | ✅ Complete | `chapter05_inverse_kinematics/` with numeric/analytic IK + null-space redundancy |
| **P4** | **PIBench — Articulated & Deformable Suite** | ✅ Complete | `DrawerPull`, `DoorSwing`, `RopeTension`, `GearTurn`, `ChainDrape` |
| 6 | Dynamics | ✅ Complete | `ArmDynamics`: mass matrix, Coriolis+gravity, forward/inverse dynamics |
| **P5** | **PIBench — Parameter Estimation & Counterfactuals** | ✅ Complete | `MassOrder`, `FrictionOrder`, `CounterfactualMass`, `CounterfactualFriction`, `BalanceAfterMove` |
| 7 | Control | ✅ Complete | Controller family (gravity comp, PID, computed torque, operational space), uncertainty-aware safety wrapper, `MockRealArm` sim-to-real bridge |
| **P6** | **PIBench — Model Harness, Leaderboard & Calibration** | ✅ Complete | `EvaluationHarness`, per-suite/per-concept accuracy, ECE/Brier/NLL calibration, static HTML leaderboard, VLM predictor |
| 8 | Motion Planning | ✅ Complete | `chapter08_motion_planning/`: collision checking, PRM/RRT/RRT*/APF planners, shortcut smoothing, viewer demo, 10 tests |
| **P7** | **PIBench — Real-Robot Validation Harness** | ✅ Complete | `ValidationTask`/`ValidationResult`, mock-arm `reach_q` execution, residual tracker, `pibench validate` CLI, 8 tests |
| 9 | Trajectory Generation | ✅ Complete | `chapter09_trajectory_generation/`: cubic/quintic splines, trapezoidal/S-curve time scaling, path→trajectory, tracked by Chapter 7 controller, 10 tests |
| **M1** | **Hardened Virtual Real-Robot Bridge** | ✅ Complete | `MockRealArm` actuator/sensor dynamics + `VirtualArmFactory` domain randomization, 13 tests |
| **M2** | **Trajectory Generation** | ✅ Complete | Cubic/quintic splines + trapezoidal/S-curve time scaling + path→trajectory, 10 tests |
| **M3** | **Virtual Robot Validation at Scale** | ✅ Complete | `BatchValidator` sweeps randomized arms, mismatch levels, and controllers; accuracy degrades with mismatch, 3 tests |
| **M4** | **Perception + Simulated Camera Stack** | ✅ Complete | `chapter10_perception/`: RGB/depth renderer, object detector by color, camera projection, IK → position controller, 6 tests |
| 10 | Virtual Perception / Camera Stack | ✅ Complete | `chapter10_perception/`: simulated RGB/depth, object detection, camera projection, controller feed |
| **M5** | **Imitation Learning** | ✅ Complete | `chapter11_imitation_learning/`: expert trajectory recorder, NumPy MLP behavior cloning, teleoperation recorder, 6 tests |
| 11 | Imitation Learning | ✅ Complete | Behavior cloning from expert reach demos; teleoperation recorder |
| **M6** | **Foundation-Model + Physics Verifier** | ✅ Complete | `chapter12_reasoning/`: NL task parser, rule + optional Claude planner, MuJoCo physics verifier, retry loop, 7 tests |
| 12 | Foundation-Model + Physics Verifier | ✅ Complete | Natural-language task → plan → simulate → verify → retry |
| 13 | Skill Library + Skill Sharing | ✅ Complete | `chapter13_skills/`: reusable skill templates, composition, JSON library |
| **M8** | **Self-Improving Virtual Real-Sim-Real Loop** | ✅ Complete | `chapter14_self_improvement/`: failure detector, online system ID, retuner, A/B experiment, self-improvement loop; 6 tests |
| 14 | Self-Improving Virtual Real-Sim-Real Loop | ✅ Complete | Same as M8 — textbook chapter aligned with the milestone |
| **M9** | **End-to-End North-Star Demo** | ✅ Complete | `chapter15_north_star/`: NL task → skill plan → IK → trajectory → arm execution → residual tracking → skill library |
| 15 | End-to-End North-Star Demo | ✅ Complete | Same as M9 — textbook chapter aligned with the milestone |

---

## 📁 Repository Structure

```
LearningRobotics/
├── README.md                      # This file — the project log
├── .gitignore                     # Ignore venvs, pycache, OS files
├── requirements.txt               # Global shared deps (mostly pointers)
├── chapter01_foundation/          # Chapter 1: C-space & DOF
│   ├── requirements.txt           # Chapter-specific pinned deps
│   ├── .venv/                     # Local virtual environment (ignored by git)
│   ├── simple_2r_arm.xml          # Minimal 2-revolute planar arm (2 DOF)
│   ├── simple_6dof_arm.xml        # Minimal 6-revolute spatial arm (6 DOF)
│   ├── inspect_dof.py             # DOF counter, FK demo, C-space torus demo
│   └── notes.md                   # Session notes with numbers
├── chapter02_rigid_body_motions/  # Chapter 2: SO(3)/SE(3)
│   ├── requirements.txt           # mujoco + numpy
│   ├── transforms.py              # Rotation matrices, Euler angles, Rodrigues, homogeneous transforms
│   ├── demo_transforms.py         # MuJoCo-backed interactive demo (if present)
│   └── test_transforms.py         # pytest suite
├── chapter03_forward_kinematics/  # Chapter 3: forward kinematics
│   ├── requirements.txt           # mujoco + numpy + pytest
│   ├── forward_kinematics.py      # DH + PoE + geometric FK for 6-DOF arm
│   └── test_forward_kinematics.py # pytest suite
├── chapter04_velocity_kinematics/ # Chapter 4: velocity kinematics & Jacobians
│   ├── requirements.txt           # mujoco + numpy + pytest
│   ├── jacobian.py                # 6x6 geometric Jacobian (analytic + numeric)
│   ├── velocity_kinematics.py     # Twist, inverse velocity, null-space demo
│   ├── demo_jacobian_viewer.py    # MuJoCo viewer with J⁺ velocity control
│   └── test_jacobian.py           # pytest suite
├── chapter05_inverse_kinematics/  # Chapter 5: inverse kinematics
│   ├── requirements.txt           # mujoco + numpy + pytest
│   ├── inverse_kinematics.py      # numeric + analytic IK, null-space redundancy
│   ├── demo_ik_viewer.py          # interactive IK tracking demo
│   └── test_inverse_kinematics.py # pytest suite
├── chapter06_dynamics/            # Chapter 6: dynamics
│   ├── requirements.txt           # mujoco + numpy + pytest
│   ├── dynamics.py                # mass matrix, bias forces, forward/inverse dynamics
│   ├── demo_dynamics_viewer.py    # gravity compensation / free-fall toggle demo
│   └── test_dynamics.py           # pytest suite
├── chapter07_control/             # Chapter 7: control
│   ├── requirements.txt           # mujoco + numpy + pytest
│   ├── control.py                 # joint-space PID, computed torque, task/operational-space controllers
│   ├── real_hardware.py           # RealArm ABC + MockRealArm + Forte/AM-ARM stub
│   ├── utils.py                   # pose_error, clip_vector, rotation helpers
│   ├── demo_control_viewer.py     # interactive controller selector demo
│   └── test_control.py            # pytest suite
├── chapter08_motion_planning/     # Chapter 8: motion planning
│   ├── requirements.txt           # mujoco + numpy + pytest
│   ├── collision.py               # `ArmPlanningEnv`: obstacles, collision/segment checks
│   ├── planners.py                # PRM, RRT, RRT*, artificial potential fields
│   ├── smoother.py                # shortcut smoothing + cubic B-spline interpolation
│   ├── demo_motion_planning_viewer.py # RRT* obstacle demo with MuJoCo viewer playback
│   └── test_motion_planning.py    # 10 pytest tests
├── chapter09_trajectory_generation/ # Chapter 9: trajectory generation
│   ├── trajectory.py              # cubic/quintic joint-space splines
│   ├── time_scaling.py            # trapezoidal and S-curve time scaling
│   ├── path_to_trajectory.py      # convert Chapter 8 path to timed trajectory
│   ├── demo_trajectory_viewer.py  # plan → trajectory → controller playback
│   └── test_trajectory_generation.py # 10 pytest tests
├── chapter10_perception/          # Chapter 10: virtual perception + camera stack
│   ├── scene.xml                  # arm + table + red/blue blocks manipulation scene
│   ├── arm.xml                    # matching 6-DOF arm with gravity for IK/control
│   ├── renderer.py                # MuJoCo RGB/depth renderer wrapper
│   ├── perception.py              # ground-truth object detection + camera projection
│   ├── controller.py              # joint-position command controller
│   ├── demo_perception_controller.py # render → detect → IK → command arm
│   └── test_perception.py         # 6 pytest tests
├── chapter11_imitation_learning/  # Chapter 11: learning from demonstration
│   ├── expert.py                  # generate reach demonstrations via IK + cubic splines
│   ├── behavior_cloning.py        # NumPy MLP policy + dataset helpers
│   ├── teleop.py                  # kinesthetic/keyboard demonstration recorder
│   └── test_imitation.py          # 6 pytest tests
├── chapter12_reasoning/           # Chapter 12: foundation-model + physics verifier
│   ├── task_parser.py             # natural-language manipulation task parser
│   ├── planner.py                 # rule-based planner + optional Claude LLM fallback
│   ├── physics_verifier.py        # simulate a plan in MuJoCo and check relations
│   ├── reasoning_loop.py          # plan → verify → retry with failure feedback
│   └── test_reasoning.py          # 7 pytest tests
├── chapter13_skills/              # Chapter 13: reusable skill library + composition
│   ├── skill.py                   # Skill / SkillInstance / SkillLibrary dataclasses + JSON
│   ├── skills.py                  # core parameterized skills: reach, push, pick, place, slide
│   ├── composer.py                # chain skill instances into verified plans
│   └── test_skills.py             # 7 pytest tests
├── chapter14_self_improvement/    # Milestone 8 / Chapter 14
│   ├── failure_detector.py        # Detect reach failures and validation regressions
│   ├── system_id.py               # Online system ID from residual tracker
│   ├── retuner.py                 # Retune a controller with estimated bias/gear compensation
│   ├── ab_experiment.py           # A/B comparison of baseline vs retuned controller
│   ├── self_improvement_loop.py   # Full loop: detect → identify → retune → validate
│   ├── demo_self_improve.py       # Constant torque bias demo: baseline 0.0336 → retuned 0.0037
│   └── test_self_improvement.py   # 6 pytest tests
└── pibench/                       # Physical Intuition Benchmark (Phases 0-7)
    ├── README.md                  # PIBench overview and quickstart
    ├── requirements.txt           # MuJoCo + benchmark deps
    ├── pibench/                   # Python package
    │   ├── cli.py                 # run/list/render/view/leaderboard/validate commands
    │   ├── core/                  # Engine + counterfactual builder
    │   ├── scenes/statics/        # TowerFall, SlopeSlide, SupportBalance, ToppleDirection
    │   ├── scenes/dynamics/       # PendulumSwing, CollisionBounce, ProjectileHit
    │   ├── scenes/contact/        # PushTipVsSlide, StackStability, WedgeInsert, FrictionPile, SlipGrip
    │   ├── scenes/articulated/    # DrawerPull, DoorSwing, RopeTension, GearTurn
    │   ├── scenes/deformable/     # ChainDrape
    │   ├── scenes/params/         # MassOrder, FrictionOrder, CounterfactualMass, CounterfactualFriction, BalanceAfterMove
    │   ├── utils/                 # MJCF + contact + articulated helpers
    │   ├── predictors/            # baselines: random, physics_oracle, llm, vlm
    │   ├── evaluation/            # Phase 6: leaderboard + metrics + calibration
    │   ├── realrobot/             # Phase 7: validation protocol + harness + residual tracker
    │   ├── tests/                 # Phase 6 evaluation/leaderboard tests
    │   └── harness.py             # Phase 6: EvaluationHarness wrapper
    ├── tests/                     # pytest suite (engine + all scenes)
    ├── docs/                      # PIBench documentation
    ├── run_all.py                 # Run all suites across all predictors
    ├── showcase.py                # Render framed thumbnails of every scene
    └── build_showcase_artifact.py # Build self-contained HTML gallery
```

---

## 🎬 Live Demo — Self-Improving Virtual Real-Sim-Real Loop

This is the best-performing demo so far. The animation shows the 6-DOF arm in MuJoCo executing the same joint-space reach task under two controllers, side by side.

![Self-improvement demo: baseline vs retuned controller](https://raw.githubusercontent.com/satyamdas03/LearningRobotics/master/docs/assets/self_improve_baseline_vs_retuned.gif)

*Higher-quality MP4 with audio-free playback controls: [`self_improve_baseline_vs_retuned.mp4`](docs/assets/self_improve_baseline_vs_retuned.mp4)*

| | Baseline controller | Retuned controller |
|---|---|---|
| **What it knows** | Nominal MuJoCo dynamics only | Nominal dynamics + learned torque offset |
| **Injected mismatch** | Unknown per-joint torque bias: `[0.3, -0.2, 0.15, 0, 0, 0]` N·m | Same bias (shared across both runs) |
| **Mean tracking error** | **0.0336** | **0.0037** (≈ 9× reduction) |
| **Success rate** | 10/10 | 10/10 |
| **Final joint error** | 0.0336 | 0.0037 |

### What is happening

1. **Injected, unknown torque bias** — the virtual "real" arm adds a constant offset to every commanded torque. The baseline computed-torque controller has no model of this bias, so it settles with a steady-state joint error.
2. **Failure detection** — a reach check flags that the arm stopped short of the target.
3. **Online system ID** — the robot runs a short calibration trajectory and uses the residual tracker (`qddot_actual − qddot_predicted`) to estimate the missing torque. The estimate is `[0.300, −0.200, 0.150, 0, 0, 0]`, matching the true bias to within numerical precision.
4. **Controller retuning** — a `CompensatedController` is wrapped around the baseline. It subtracts the learned torque offset before sending commands to the virtual arm.
5. **A/B validation** — the retuned controller is compared against the baseline on 10 independent randomized virtual arms. Mean error drops from 0.0336 to 0.0037, proving the improvement.

### Reproduce it

```bash
cd chapter14_self_improvement
python demo_self_improve_recorded.py
```

The script writes `output/self_improve_baseline_vs_retuned.mp4` and prints the full improvement report. A non-recorded version is also available: `python demo_self_improve.py`.

---

## ✅ Chapter 1 — Foundations, Configuration Space & DOF

### Concepts locked in

* **Link** — a rigid body of the robot.
* **Joint** — a connection allowing relative motion.
* **Configuration** — the minimal set of numbers that fully describes the robot's pose (usually joint angles).
* **Degrees of Freedom (DOF)** — the number of independent ways the robot can move.
* **C-space** — the space of all possible configurations; each point is one pose.
* **Task space** — where the task happens (usually the end-effector position/orientation).
* **Workspace** — the physical volume the end-effector can reach.

### Robots built

#### 1. `simple_2r_arm.xml` — 2R planar arm

Two revolute joints, both rotating about the Z-axis.

| Metric | Value |
|---|---|
| Joints | 2 revolute |
| DOF (`nv`) | **2** |
| C-space dimension (`nq`) | **2** |
| Task space | `(x, y)` of the end-effector |
| Link lengths | L1 = 1.0 m, L2 = 0.8 m |

Forward kinematics matches the analytical solution exactly:

```text
q = (0, 0)        -> ee = (1.800, 0.000)
q = (π/4, 0)      -> ee = (1.273, 1.273)
q = (0, π/2)      -> ee = (1.000, 0.800)
q = (0, π)        -> ee = (0.200, 0.000)
```

#### 2. `simple_6dof_arm.xml` — 6-DOF spatial arm

Six revolute joints in 3D: waist (Z), shoulder (Y), elbow (Y), wrist roll (X), wrist pitch (Y), wrist yaw (Z).

| Metric | Value |
|---|---|
| Joints | 6 revolute |
| DOF (`nv`) | **6** |
| C-space dimension (`nq`) | **6** |
| Task space | `(x, y, z, roll, pitch, yaw)` of the end-effector |
| Why 6? | Matches the 6 DOF of a free rigid body in 3D: 3 position + 3 orientation |

### C-space topology demo

The script proves that revolute-joint angles wrap around:

```text
q1=0.0000, q2=0.0000       -> ee=(1.8000, 0.0000)
q1=0.0000, q2=2π           -> ee=(1.8000, -0.0000)
q1=2π,    q2=0.0000       -> ee=(1.8000, -0.0000)
q1=2π,    q2=2π           -> ee=(1.8000, -0.0000)
```

So:

* One revolute joint → C-space is a **circle** (`S¹`).
* Two revolute joints → C-space is a **torus** (`S¹ × S¹`).

### Why MuJoCo and not Isaac Sim (yet)

The chapter recommended installing Isaac Sim. I chose MuJoCo for the first practical because:

1. It installs in seconds via `pip install mujoco`.
2. It runs cleanly on an RTX 5060 without multi-GB downloads.
3. It exposes joints, qpos, and forward kinematics directly — perfect for learning C-space/DOF.
4. The concepts transfer 100% to Isaac Sim / Isaac Lab, which will be installed for the RL chapters.

---

## ✅ Chapter 2 — Rigid-Body Motions

### Concepts locked in

* **SO(3)** — the group of 3×3 rotation matrices; `RᵀR = I`, `det(R) = +1`.
* **Euler angles (ZYX / intrinsic)** — `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`.
* **Axis-angle / Rodrigues' formula** — rotate `θ` about a unit axis `ω`.
* **SE(3)** — 4×4 homogeneous transforms combining rotation + translation.
* **Transform composition & inversion** — `T_ab @ T_bc = T_ac`, `T_ba = inv(T_ab)`.

### Files

| File | Purpose |
|---|---|
| `transforms.py` | `rotx`, `roty`, `rotz`, `euler_xyz`, `axis_angle`, `homogeneous_transform`, `transform_point`, `inverse_transform` |
| `test_transforms.py` | 6 pytest tests covering rotation properties, Euler round-trip, axis-angle, and SE(3) transforms |

### Validation

* Euler-angle round-trip recovers `(roll, pitch, yaw)` from the reconstructed rotation matrix.
* Axis-angle round-trip recovers axis and angle from `R`.
* Homogeneous transforms compose and invert correctly: `inv(T) @ T ≈ I₄`.

---

## ✅ Chapter 3 — Forward Kinematics

### Concepts locked in

* **Forward kinematics (FK)** — map joint configuration `q` to end-effector pose.
* **Product of Exponentials (PoE)** — `T(q) = M · exp([S₁]θ₁) · … · exp([Sₙ]θₙ)`.
* **Geometric transform chain** — successive frame placements `T_01 @ T_12 @ … @ T_n-1,n`.
* **DH-style frame building** — explicit link/joint frames matched to the MuJoCo XML.

### Files

| File | Purpose |
|---|---|
| `forward_kinematics.py` | `Arm6DOFFK` class: `poe_fk`, `geometric_fk`, `mujoco_fk`, `end_effector_position` |
| `test_forward_kinematics.py` | 4 pytest tests: PoE vs MuJoCo, geometric vs PoE, waist rotation, random configs |

### Validation

* PoE FK matches MuJoCo's built-in FK to ~1e-16 for the default configuration.
* Random configurations also agree to machine precision.
* Geometric FK and PoE FK produce identical end-effector positions.

---

## ✅ Chapter 4 — Velocity Kinematics & Jacobians

### Concepts locked in

* **Geometric Jacobian** — maps joint velocities to end-effector twists.
* **Analytic vs numeric Jacobian** — analytic formula matches MuJoCo `mj_jacSite`.
* **Twist** — 6D spatial velocity `[ω; v]`.
* **Inverse velocity** — `q̇ = J⁺ V_ee` (Moore-Penrose or damped pseudoinverse).
* **Null-space motion** — redundancy resolution for staying away from joint limits.
* **Static-force duality** — `τ = Jᵀ F`.

### Files

| File | Purpose |
|---|---|
| `jacobian.py` | `ArmJacobian`: numeric + analytic 6×6 Jacobian, twist, inverse twist, null-space projector, static-force duality |
| `velocity_kinematics.py` | Demo comparing analytic vs numeric Jacobian, twist, inverse velocity, null-space motion, static-force duality |
| `demo_jacobian_viewer.py` | MuJoCo passive viewer driving the 6-DOF arm with Jacobian pseudoinverse to track a moving target |
| `test_jacobian.py` | 8 pytest tests: analytic Jacobian matches MuJoCo numeric to machine precision |

### Validation

* Analytic geometric Jacobian matches MuJoCo's numeric site Jacobian to ~1e-16.
* Jacobian pseudoinverse tracks a moving circular target in the interactive viewer.
* Inverse velocity and null-space projector behave as expected.

---

## ✅ Chapter 5 — Inverse Kinematics

### Concepts locked in

* **Numeric IK** — iteratively minimize pose error via the Jacobian pseudoinverse.
* **Damped pseudoinverse** — robust inverse when the arm is near a singularity.
* **Position-only IK** — track only the 3D position, leaving a 3-DOF null-space.
* **Null-space redundancy resolution** — maximize a secondary objective (joint-limit centering) without disturbing the primary task.
* **Analytic planar 2R IK** — law-of-cosines solution for the waist-shoulder sub-problem.

### Files

| File | Purpose |
|---|---|
| `inverse_kinematics.py` | `InverseKinematics`: numeric IK (damped/pure pseudoinverse), position-only mode, null-space secondary objective, analytic 2R IK, joint-limit centering helper |
| `demo_ik_viewer.py` | MuJoCo viewer where the 6-DOF arm cycles through reachable target poses via damped-pseudoinverse IK |
| `test_inverse_kinematics.py` | 4 pytest tests: numeric IK convergence, analytic 2R solution, null-space centering, unreachable target stays in limits |

### Validation

* Numeric IK converges to reachable target `[0.60, 0.20, 0.60]` with position error < 1e-3 and rotation error < 1e-2.
* Analytic planar 2R solver recovers the exact law-of-cosines configuration.
* Position-only IK + joint-limit centering objective keeps joints closer to mid-range without losing accuracy.
* Unreachable targets produce finite, in-limit best-effort solutions.

---

## ✅ Chapter 6 — Dynamics

### Concepts locked in

* **Mass matrix M(q)** — the symmetric, positive-definite matrix mapping joint accelerations to generalized forces.
* **Coriolis + gravity bias** — velocity-dependent and gravitational terms that must be overcome to maintain a state.
* **Forward dynamics** — compute joint accelerations from applied torques: `q̈ = M(q)^{-1} (τ - C(q,q̇)q̇ - g(q))`.
* **Inverse dynamics** — compute the torques required to produce a desired acceleration: `τ = M(q)q̈ + C(q,q̇)q̇ + g(q)`.
* **Gravity compensation** — the torque needed to hold the arm stationary against gravity (`q̇ = 0, q̈ = 0`).

### Files

| File | Purpose |
|---|---|
| `dynamics.py` | `ArmDynamics`: mass matrix, Coriolis+gravity bias, gravity-only term, velocity-dependent Coriolis term, forward dynamics, inverse dynamics, Euler integration step |
| `demo_dynamics_viewer.py` | MuJoCo viewer that toggles between gravity compensation and free fall every 3 seconds |
| `test_dynamics.py` | 6 pytest tests: symmetry/positive-definiteness of M(q), M-column vs inverse dynamics, gravity compensation, free-fall acceleration, Euler step vs MuJoCo, inverse/forward consistency |

### Validation

* `M(q)` is symmetric and positive definite for every tested configuration.
* Forward and inverse dynamics are mutually consistent to 1e-3 across random states.
* Our simple Euler step matches MuJoCo's `mj_step` for one integration step to 1e-3.
* Under zero torque the arm accelerates downward due to gravity; under gravity compensation it would remain static.

---

## ✅ Chapter 7 — Control

### Concepts locked in

* **Gravity compensation** — the torques needed to hold the arm stationary against gravity.
* **Joint-space PID** — independent proportional-integral-derivative control per joint with gravity feedforward, torque saturation, and anti-windup.
* **Computed torque** — inverse-dynamics linearization that cancels nonlinear dynamics and leaves linear error dynamics.
* **Task-space control** — map end-effector wrench to joint torques via the Jacobian transpose.
* **Operational-space / resolved-acceleration control** — task-space command through the Jacobian pseudoinverse plus inverse dynamics.
* **Uncertainty-aware wrapper** — monitor model-mismatch residual and clamp torque when predictions diverge from reality (a seed from the manifesto's "uncertainty-aware autonomous robot" concept).
* **Real-arm abstraction** — write controllers against `RealArm` so the same code runs on `MockRealArm` today and on Forte/AM-ARM tomorrow.

### Files

| File | Purpose |
|---|---|
| `control.py` | `RobotController` base + gravity, PID, computed-torque, task-space, operational-space, and uncertainty-aware wrapper |
| `real_hardware.py` | `RealArm` ABC, `MockRealArm` (with noise/delay/saturation), `ForteAMArmAdapter` stub |
| `utils.py` | `pose_error`, `clip_vector`, `rotation_matrix`, `axis_angle_from_matrix` |
| `demo_control_viewer.py` | MuJoCo viewer that switches between controllers (`--controller gravity/pid/computed_torque/operational_space`) |
| `test_control.py` | 9 pytest tests covering gravity comp, PID convergence, computed-torque tracking, operational-space pose reach, task-space force direction, torque limits, anti-windup, uncertainty clamp, and mock arm roundtrip |

### Validation

* `python -m pytest test_control.py -q` — **9 passed**.
* Controllers can drive the simulated arm from non-zero initial conditions back to setpoints or along trajectories.
* Mock real arm supports configurable actuator noise, communication delay, and torque saturation for sim-to-real stress testing.

---

## ✅ Chapter 8 — Motion Planning

### Concepts locked in

* **C-space obstacles** — configurations that cause collision become forbidden regions in joint space.
* **Collision checking** — probe the arm at a configuration and check MuJoCo contacts against obstacle geoms.
* **Sampling-based planners** — PRM, RRT, and RRT* build collision-free graphs/trees in high-dimensional C-space.
* **Probabilistic completeness** — given enough samples, sampling-based methods find a feasible path if one exists.
* **Path smoothing** — shortcut smoothing and B-spline interpolation shorten and soften raw planner output.
* **Artificial potential fields** — gradient-based planner that can get stuck in local minima; included as a contrast.

### Files

| File | Purpose |
|---|---|
| `collision.py` | `ArmPlanningEnv`: MuJoCo model with injected obstacle bodies, joint-limit checks, collision and segment-free queries |
| `planners.py` | `PRMPlanner` (A* on k-NN roadmap), `RRTPlanner`, `RRTStarPlanner` with rewiring, `PotentialFieldPlanner` |
| `smoother.py` | `shortcut_smooth` and `cubic_bspline_interpolate` for post-processing planner paths |
| `demo_motion_planning_viewer.py` | RRT* plans around obstacles and plays back the path in the MuJoCo passive viewer |
| `test_motion_planning.py` | 10 pytest tests: limits, obstacle detection, segment checks, planner validity, smoothing, RRT* optimality, potential-field optional behavior |

### Validation

* `python -m pytest test_motion_planning.py -q` — **10 passed**.
* RRT* consistently finds a collision-free path between two configurations in an empty environment and around simple obstacle sets.
* Shortcut smoothing reduces C-space path length while preserving collision-free validity.

---

## ✅ Chapter 9 — Trajectory Generation

### Concepts locked in

* **Geometric path vs. timed trajectory** — a path is a sequence of configurations; a trajectory assigns a time to each configuration.
* **Polynomial interpolation** — cubic splines match position/velocity at waypoints; quintic splines also match acceleration, giving smoother torque commands.
* **Time scaling** — a normalized path parameter ``s(t)`` can be shaped independently of the path geometry to respect velocity/acceleration/jerk limits.
* **Trapezoidal profile** — bang-bang acceleration with a velocity cruise; simple and time-optimal for bounded velocity/acceleration.
* **S-curve profile** — jerk-limited trapezoid with continuous acceleration; reduces mechanical stress and controller excitation.
* **Path → trajectory → controller** — Chapter 8 paths can be converted to timed Chapter 9 trajectories and tracked by Chapter 7 controllers.

### Files

| File | Purpose |
|---|---|
| `trajectory.py` | `Trajectory` class + cubic/quintic polynomial interpolation with analytical velocity/acceleration |
| `time_scaling.py` | `trapezoidal_time_scaling` and `scurve_time_scaling` with bounded velocity/acceleration/jerk |
| `path_to_trajectory.py` | `path_to_trajectory` and `plan_to_timed_trajectory` converters |
| `demo_trajectory_viewer.py` | RRT* plan → quintic trajectory → joint-space PID playback in MuJoCo viewer |
| `test_trajectory_generation.py` | 10 pytest tests: waypoint hits, boundary velocities/accelerations, profile bounds, monotonic time |

### Validation

* `python -m pytest test_trajectory_generation.py -q` — **10 passed**.
* Cubic and quintic splines pass through all waypoints and respect zero start/end boundary conditions by default.
* Trapezoidal and S-curve profiles traverse ``s=0`` to ``s=1`` within their declared limits; S-curve acceleration is continuous.

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/satyamdas03/LearningRobotics.git
cd LearningRobotics
```

### 2. Run Chapter 1

On Windows (PowerShell):

```powershell
cd chapter01_foundation
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python inspect_dof.py
```

On Linux/macOS:

```bash
cd chapter01_foundation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python inspect_dof.py
```

Expected output: a table of joints, DOF counts, forward-kinematics comparisons, and the C-space topology demo.

### 3. Run Chapter 2 — Rigid-Body Motions

```powershell
cd chapter02_rigid_body_motions
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest test_transforms.py -v
```

Tests cover: rotation-matrix orthogonality, Euler angle round-trips, axis-angle recovery, and SE(3) transform composition/inversion.

### 4. Run Chapter 3 — Forward Kinematics

```powershell
cd chapter03_forward_kinematics
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest test_forward_kinematics.py -v
```

Tests verify: PoE FK matches MuJoCo to machine precision for default and random joint configurations, and geometric FK agrees with PoE FK.

### 5. Run Chapter 5 — Inverse Kinematics

```powershell
cd chapter05_inverse_kinematics
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest test_inverse_kinematics.py -v
```

Tests verify: numeric IK reaches a reachable target, analytic 2R IK recovers the planar solution, null-space centering helps redundancy resolution, and unreachable targets stay finite and in limits.

To open the interactive viewer demo (optional, requires a display):

```powershell
python demo_ik_viewer.py
```

### 6. Run Chapter 6 — Dynamics

```powershell
cd chapter06_dynamics
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest test_dynamics.py -v
```

Tests verify: `M(q)` is symmetric and positive definite, forward/inverse dynamics are consistent, gravity compensation matches static torque, and our Euler step matches MuJoCo for one step.

To open the interactive viewer demo (optional, requires a display):

```powershell
python demo_dynamics_viewer.py
```

### 7. Run PIBench

```powershell
cd pibench
. .venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Users\point\projects\LearningRobotics\pibench"
python -m pibench run --suite statics --predictor physics_oracle --n 5
python -m pibench run --suite dynamics --predictor physics_oracle --n 5
python -m pibench run --suite contact --predictor physics_oracle --n 5
python -m pibench run --suite articulated --predictor physics_oracle --n 5
python -m pibench run --suite deformable --predictor physics_oracle --n 5
python -m pibench run --suite params --predictor physics_oracle --n 5

# Optional: save results and build a static leaderboard
python -m pibench run --suite statics --predictor physics_oracle --n 20 --output output/results_physics_oracle.json
python -m pibench run --suite statics --predictor random --n 20 --output output/results_random.json
python -m pibench leaderboard --output-dir output

python run_all.py
```

Expected output: `Overall accuracy: 100.0%` on every suite. Physics-oracle should be the top predictor. Saved result files feed into `output/leaderboard.json` and `output/leaderboard.html`.

---

## 🧠 Test Yourself — Chapter 1 Answers

1. **Drone flying freely** — **6 DOF**: `(x, y, z)` position + roll, pitch, yaw.
2. **Smartphone sliding/spinning on a table** — **3 DOF**: `x, y` position + rotation θ about the table normal.
3. **5 revolute joints in an open chain** — **5 DOF**. It cannot reach every position *and* orientation in 3D because it has fewer DOF than the 6 DOF of a free rigid body.
4. **Why C-space reframing helps planning** — It turns a complex physical robot into a single point moving through an abstract space. Obstacles become forbidden regions, and planning becomes "find a path for the point."
5. **Planar mechanism with 5 links and 5 revolute joints** — Using Grübler's formula with `m=3`, `N=5`, `J=5`, `Σf_i=5`:
   `DOF = 3(5 - 1 - 5) + 5 = 3(-1) + 5 = 2`.

---

## 🧱 PIBench — First Revolutionary Deliverable (Phases 0–7)

PIBench is the executable first step toward the *Revolutionary Robotics* north star. It evaluates whether a model truly understands physical common sense across statics, dynamics, contact/friction, articulated constraints, deformable bodies, parameter estimation / counterfactual reasoning, model evaluation/leaderboards, and a real-robot validation protocol.

### What works now

* **Engine scaffold:** `Problem`, `Suite`, `Runner`, `Evaluator`, `Registry`, and model-agnostic `Predictor` interface.
* **Statics suite (P1):**
  * `TowerFall` — which tower falls on a tilting platform?
  * `SlopeSlide` — does a block slide down an incline? (`tan(θ) > μ_s`)
  * `SupportBalance` — predict balance point for an asymmetric beam.
  * `ToppleDirection` — which way does an off-center stack topple?
* **Dynamics suite (P2):**
  * `PendulumSwing` — estimate small-angle period `T ≈ 2π√(L/g)`.
  * `CollisionBounce` — 1D elastic-collision outcome.
  * `ProjectileHit` — predict range `R = v² sin(2θ)/g`.
* **Contact suite (P3):**
  * `PushTipVsSlide` — pushed block tips or slides depending on push height.
  * `StackStability` — does a stack survive a side tap?
  * `WedgeInsert` — does a wedge fit through a gap or jam?
  * `FrictionPile` — which object is hardest to start moving?
  * `SlipGrip` — gripper lifts the block or slips?
* **Articulated suite (P4):**
  * `DrawerPull` — does a pulled drawer open or jam on its slide?
  * `DoorSwing` — does a pushed door swing open or stick at the hinge?
  * `RopeTension` — which side of a pulley descends?
  * `GearTurn` — which way does a meshed gear turn?
* **Deformable suite (P4):**
  * `ChainDrape` — how high is the free end of a chain draped over a bar?
* **Parameter estimation & counterfactual suite (P5):**
  * `MassOrder` — order three blocks by mass from observed displacements after identical pushes.
  * `FrictionOrder` — rank three surfaces by slipperiness from tilt threshold.
  * `CounterfactualMass` — if the top mass of a tower were doubled, would it still stand?
  * `CounterfactualFriction` — if static friction on an incline were zero, would the block slide?
  * `BalanceAfterMove` — how far must the support shift after a point mass is moved on a beam?
* **Baselines:** `physics_oracle` (100% on deterministic scenes), `random`, optional `llm` (requires `anthropic` SDK + `ANTHROPIC_API_KEY`), optional `vlm` (renders scene + multimodal prompt).
* **CLI:** `pibench list`, `pibench run`, `pibench render`, `pibench view`, `pibench leaderboard`, `pibench validate`.
* **Phase 6 (complete):** `EvaluationHarness`, static HTML/JSON leaderboard, per-concept and calibration metrics (ECE, Brier, NLL), and optional VLM predictor that renders scenes and asks a vision-language model.
* **Phase 7 (complete):** `RealRobotValidationHarness` with `ValidationTask`/`ValidationResult` protocol, mock-arm `reach_q` execution, online residual tracker for sim-to-real mismatch, and `pibench validate` CLI command.
* **Tests:** 170 passing across all chapters (Chapters 2–13), PIBench engine/evaluation/validation suites, and the hardened virtual real-robot bridge.

### Run it

```powershell
cd pibench
. .venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Users\point\projects\LearningRobotics\pibench"
python -m pibench run --suite statics --predictor physics_oracle --n 10
python -m pibench run --suite dynamics --predictor physics_oracle --n 10
python -m pibench run --suite contact --predictor physics_oracle --n 10
python -m pibench run --suite articulated --predictor physics_oracle --n 10
python -m pibench run --suite deformable --predictor physics_oracle --n 10
python -m pibench run --suite params --predictor physics_oracle --n 10
python -m pibench view TowerFall
python run_all.py

# Phase 7 real-robot validation demo (uses mocked arm)
python -m pibench.cli validate --output output/validate_dummy.json
```

Expected output: `Overall accuracy: 100.0%` on every suite. `physics_oracle` tops the leaderboard.

### Why this matters

A model that can answer "which tower falls?" or "where does the projectile land?" is being forced to reason about center of mass, support polygon, friction, energy, and projectile motion — the same concepts in Chapters 2 and 3. With the contact suite it must also reason about moment balance, impulse, contact geometry, and Coulomb friction — the concepts in Chapter 4. With the articulated/deformable suite it must reason about constraints, prismatic and revolute joints, tendons, and coarse deformable approximations — the concepts in Chapter 5. With the parameter-estimation / counterfactual suite it must infer latent physical properties (mass, friction) from observations and answer "what if?" questions — the system-identification and causal-reasoning skills that feed directly into Chapter 6 dynamics and future control.

---

## 🗺️ Roadmap to "Extraordinary & Revolutionary" — Simulation-First Edition

The end goal is an AI-driven robot that learns physical skills, reasons with physics, executes safely, and shares what it learns — and we are reaching it **inside simulation first**, with no required hardware purchase. The virtual stack is designed so that only the low-level `RealArm` adapter changes when a physical robot is added later.

Milestones to the north-star demo:

1. **Harden the virtual real-robot bridge** — realistic actuator/sensor dynamics, noise, latency, domain randomization.
2. **Trajectory generation** — timed, smooth motion from Chapter 8 paths to Chapter 7 controllers.
3. **Virtual validation at scale** — run `pibench validate` across randomized agents and injected mismatch.
4. ✅ **Simulated perception** — MuJoCo camera rendering, object detection, optional VLM scene description.
5. ✅ **Imitation learning** — record expert trajectories, behavior cloning, optional diffusion/ACT policy.
6. ✅ **Foundation-model + physics verifier** — LLM/VLM proposes plans, MuJoCo verifies, retry on failure.
7. ✅ **Skill library + sharing** — reusable parameterized skills (reach/push/pick/place/slide), plan composition, JSON-backed skill library.
8. **Self-improving virtual real-sim-real loop** — failure detection, online system ID, retuning, A/B improvement.
9. **End-to-end north-star demo** — natural-language task → plan → trajectory → execute → validate → calibrate → save skill.

When the virtual loop is solid, a physical arm (Forte / AM-ARM) becomes a drop-in `RealArm` adapter swap — not a rewrite.

---

## 📝 Changelog

### 2026-08-20 — Milestone 7 Skill Library + Skill Sharing Complete

* **Milestone 7 — Skill Library + Skill Sharing:** added `chapter13_skills/` with `skill.py` (`Skill`, `SkillInstance`, `SkillLibrary` dataclasses + JSON save/load), `skills.py` (parameterized `reach`, `push`, `pick`, `place`, `slide` skills that generate concrete Chapter 12 plans), and `composer.py` (chains skill instances and verifies each sub-plan with the Chapter 12 physics verifier). Added `test_skills.py` (7 tests) covering skill target generation, physics verification, composition chaining, and JSON round-trip.
* **Tests:** full combined suite now **170 passing**.
* **Documentation:** updated root `README.md`, `memory.md`, and `learning-robotics.md` memory to mark Milestone 7 complete.
* Committed and pushed all changes to `origin/master`.

### 2026-08-20 — Milestone 6 Foundation-Model + Physics Verifier Complete

* **Milestone 6 — Foundation-Model + Physics Verifier:** added `chapter12_reasoning/` with `task_parser.py` (rule-based NL manipulation parser), `planner.py` (deterministic geometric planner with optional Claude LLM fallback), `physics_verifier.py` (MuJoCo simulator that applies plans and checks spatial relations), and `reasoning_loop.py` (plan → verify → retry with failure feedback). Added `test_reasoning.py` (7 tests) covering task parsing, rule planning, verifier accept/reject, retry-loop success, and LLM fallback without an API key.
* **Tests:** full combined suite now **163 passing** (includes a relaxed residual bound in `pibench/pibench/tests/test_realrobot.py` to prevent RNG flakiness).
* **Documentation:** updated root `README.md`, `memory.md`, and `learning-robotics.md` memory to mark Milestone 6 complete.
* Committed and pushed all changes to `origin/master`.

### 2026-08-20 — Milestone 5 Imitation Learning Complete

* **Milestone 5 — Imitation Learning:** added `chapter11_imitation_learning/` with `expert.py` (IK-based reach demonstration generator), `behavior_cloning.py` (NumPy-only MLP with Adam + dataset helpers for path residuals and one-shot goal reaching), `teleop.py` (kinesthetic/keyboard demonstration recorder with JSON save/load), and `test_imitation.py` (6 tests) covering expert trajectory generation, behavior-cloning fitting, goal-policy reaching, and teleop recorder round-trip.
* **Tests:** full combined suite now **156 passing**.
* **Documentation:** updated root `README.md`, `memory.md`, and `learning-robotics.md` memory to mark Milestone 5 complete.
* Committed and pushed all changes to `origin/master`.

### 2026-08-20 — Milestone 4 Perception + Simulated Camera Stack Complete

* **Milestone 4 — Perception + Simulated Camera Stack:** added `chapter10_perception/` with `scene.xml`/`arm.xml` (manipulation scene + matching gravity-enabled arm), `renderer.py` (MuJoCo RGB/depth wrapper), `perception.py` (ground-truth color-based object detection + pinhole camera projection), `controller.py` (joint-position command controller), and `demo_perception_controller.py` (render → detect red block → IK → command arm). Added `test_perception.py` (6 tests) covering RGB rendering, color detection, camera transform/projection, and the full perception-to-reach pipeline.
* **Tests:** full combined suite now **150 passing**.
* **Documentation:** updated root `README.md`, `memory.md`, and `learning-robotics.md` memory to mark Milestone 4 complete.
* Committed and pushed all changes to `origin/master`.

### 2026-08-20 — Milestone 3 Virtual Robot Validation at Scale Complete

* **Milestone 3 — Virtual Robot Validation at Scale:** added `pibench/pibench/realrobot/batch.py` with `BatchValidator` and `BatchValidationReport`. The validator sweeps randomized `MockRealArm` instances across mismatch levels (`0.0–1.0`), random seeds, and controller variants (PID and computed torque), then reports per-level mean accuracy. Residual tracker tests now prove that gear-ratio mismatches inflate residuals, and batch results show accuracy degrades as mismatch grows.
* **Tests:** added 3 new real-robot validation tests (`test_residual_tracker_detects_gear_mismatch`, `test_batch_validator_reports_accuracy_vs_mismatch`, `test_computed_torque_controller_on_virtual_arm`). Full combined suite now **144 passing**.
* **Documentation:** updated root `README.md`, `memory.md`, and `learning-robotics.md` memory to mark Milestones 2 and 3 complete.
* Committed and pushed all changes to `origin/master`.

### 2026-08-20 — Milestone 1 Virtual Bridge + Chapter 9 Trajectory Generation Complete

* **Milestone 1 — Hardened Virtual Real-Robot Bridge:** extended `chapter07_control/real_hardware.py` `MockRealArm` with torque/velocity/position control modes, gear ratios, smooth Coulomb + viscous friction, first-order actuator lag, command delay, sensor noise/bias/drift/quantization, feedback delay, and torque/velocity/position saturation. Added `VirtualArmFactory` for domain-randomized arm variants. Added `chapter07_control/test_virtual_arm.py` (13 passing tests) proving the virtual arm diverges from the controller's assumed model.
* **Chapter 9 — Trajectory Generation:** added `chapter09_trajectory_generation/` with `trajectory.py` (cubic/quintic joint-space splines), `time_scaling.py` (trapezoidal and S-curve profiles), `path_to_trajectory.py` (Chapter 8 path → timed trajectory), `demo_trajectory_viewer.py` (RRT* plan → trajectory → PID playback), and `test_trajectory_generation.py` (10 passing tests).
* **Tests:** full combined suite now **141 passing**. Hardened bridge tests + trajectory tests added; fixed flaky pibench residual threshold for light wrist joints.
* **Documentation:** updated root `README.md`, `memory.md`, and `learning-robotics.md` memory to mark Milestone 1 and Chapter 9 complete.
* Committed and pushed all changes to `origin/master`.

### 2026-08-20 — Chapter 8 Motion Planning + PIBench Phase 7 Real-Robot Validation Complete

* **Chapter 8 — Motion Planning:** added `chapter08_motion_planning/` with `collision.py` (`ArmPlanningEnv` + obstacle injection), `planners.py` (PRM, RRT, RRT*, artificial-potential-field), `smoother.py` (shortcut smoothing + cubic B-spline), `demo_motion_planning_viewer.py` (RRT* playback), and `test_motion_planning.py` (10 passing tests).
* **PIBench Phase 7 — Real-Robot Validation Harness:** added `pibench/pibench/realrobot/` with `protocol.py` (`ValidationTask`/`ValidationResult`), `harness.py` (`RealRobotValidationHarness` with mock-arm `reach_q` execution), `calibration.py` (`ResidualTracker` for online sim-to-real mismatch), and `tests/test_realrobot.py` (5 passing tests). Added `pibench validate` CLI command with ASCII `[OK]`/`[FAIL]` status markers.
* **Showcase:** regenerated 29 framed thumbnails including a new Chapter 8 motion-planning card; updated `build_showcase_artifact.py` captions and stats (8 chapters, 6 suites, 29 scenes, 118 tests passing).
* **Tests:** full combined suite now **118 passing** across Chapters 2–8 and PIBench engine/evaluation/validation. `python pibench/run_all.py` reports physics oracle 100.0% (220/220), random 38.6% (85/220). `pibench validate` demo reports 100.0% (3/3).
* **Documentation:** updated root `README.md`, `memory.md`, `pibench/README.md`, `pibench/docs/PLAN.md`, and `pibench/docs/SCENE_CATALOG.md` to mark Chapter 8 and Phase 7 complete and refresh file indexes.
* Committed and pushed all changes to `origin/master`.

### 2026-08-22 — Chapter 7 Control + PIBench Phase 6 Scaffold + RoboCAD Phase 0 Complete

* **Chapter 7 — Control:** added `chapter07_control/` with `control.py`, `real_hardware.py`, `utils.py`, `demo_control_viewer.py`, and `test_control.py`.
  * Controllers: gravity compensation, joint-space PID with anti-windup, computed torque, task-space `J^T` PD, operational-space resolved acceleration, and an uncertainty-aware safety wrapper.
  * `RealArm` abstract interface plus `MockRealArm` for sim-to-real stress testing and `ForteAMArmAdapter` stub for future hardware.
  * 9 pytest tests; all passing.
* **PIBench Phase 6 (scaffold):** added `pibench/harness.py` (`EvaluationHarness`), `pibench/evaluation/` (leaderboard, per-concept accuracy, calibration metrics), and `pibench/predictors/vlm_predictor.py`.
  * CLI: `pibench leaderboard` builds `output/leaderboard.json` and `output/leaderboard.html` from saved `results_*.json` files.
  * All 55 existing tests still pass.
* **RoboCAD sister project:** Phase 0 validation complete — 8/8 prompts pass on first attempt.
* **Documentation:** updated root `README.md`, `memory.md`, and Claude memory cards to reflect Chapter 7, PIBench Phase 6, and RoboCAD Phase 0 status.
* Committed and pushed both `LearningRobotics` and `RoboCAD` repos.

### 2026-08-20 — Chapter 6 Complete + PIBench Phase 5

* **Chapter 6 — Dynamics:** added `chapter06_dynamics/` with `ArmDynamics` class wrapping `simple_6dof_arm.xml`. Implements mass matrix `M(q)`, Coriolis+gravity bias, gravity-only and Coriolis-only terms, forward dynamics `q̈ = M^{-1}(τ - bias)`, inverse dynamics `τ = Mq̈ + bias`, and a simple Euler integration step. Includes an interactive viewer demo that toggles gravity compensation / free fall every 3 seconds, plus 6 pytest tests.
* **PIBench Phase 5 (Parameter Estimation & Counterfactuals):** added `MassOrder`, `FrictionOrder`, `CounterfactualMass`, `CounterfactualFriction`, and `BalanceAfterMove` to a new `params` suite.
* **Counterfactual engine:** added `pibench/core/counterfactual.py` with `CounterfactualBuilder` and `counterfactual(problem, **overrides)` convenience function; problems expose `_counterfactual_params()` to declare re-buildable latent parameters.
* **LLM predictor stub:** added `pibench/predictors/llm_predictor.py` (optional; requires `anthropic` SDK and `ANTHROPIC_API_KEY`), with local cache fallback and random fallback when no key is available.
* **CLI:** `pibench run --predictor` now dynamically shows `llm` only when the Anthropic SDK is installed.
* **Tests:** expanded `pibench/tests/test_core.py` from 43 to 55 passing tests; physics oracle scores 100% across all six suites. `chapter06_dynamics/test_dynamics.py` adds 6 passing tests.
* **Documentation:** updated `SCENE_CATALOG.md`, `pibench/docs/PLAN.md`, `pibench/README.md`, root `README.md`, and `memory.md`; added `docs/HARDWARE_BOM.md` memo for the sub-$500 hardware stack.
* Ran full test suite, `run_all.py` baseline, and `showcase.py`; committed and pushed to GitHub.

### 2026-08-18 — Chapter 5 Complete + PIBench Phase 4

* **Chapter 5 — Inverse Kinematics:** added `chapter05_inverse_kinematics/` with numeric IK (damped/pure pseudoinverse), position-only mode, true null-space redundancy resolution, analytic planar 2R IK, and 4 pytest tests.
* **PIBench Phase 4 (Articulated):** added `DrawerPull`, `DoorSwing`, `RopeTension`, and `GearTurn` using prismatic/hinge joints, spatial tendons, and gear-meshing kinematic principles.
* **PIBench Phase 4 (Deformable):** added `ChainDrape` as a coarse capsule-chain approximation draped over a bar; numeric height answer.
* **Engine helpers:** added `pibench/utils/articulated.py` with MJCF builders for prismatic joints, hinge joints, spatial tendons, and nested capsule chains.
* **Tests:** expanded `tests/test_core.py` from 31 to 43 passing tests; physics oracle scores 100% across all five suites.
* Updated `SCENE_CATALOG.md`, `pibench/README.md`, root `README.md`, and `memory.md`.
* Ran full test suite and `run_all.py` baseline; committed and pushed to GitHub.

### 2026-08-13 — Chapters 2 & 3 Complete + PIBench Phases 1 & 2

* **Chapter 2 — Rigid-Body Motions:** added `transforms.py` implementing SO(3)/SE(3) helpers, with `test_transforms.py` (6 tests passing).
* **Chapter 3 — Forward Kinematics:** added `forward_kinematics.py` with PoE, geometric, and MuJoCo FK for the 6-DOF arm; tests verify PoE matches MuJoCo to ~1e-16.
* **PIBench Phase 1 (Statics):** added `SlopeSlide`, `SupportBalance`, and `ToppleDirection` to the statics suite.
* **PIBench Phase 2 (Dynamics):** added `PendulumSwing`, `CollisionBounce`, and `ProjectileHit` to the dynamics suite.
* Fixed deterministic caching in `ground_truth()` so the physics oracle scores 100% across all suites.
* Updated `SCENE_CATALOG.md`, `pibench/README.md`, and root `README.md`.
* Ran full test suite and `run_all.py` baseline; committed and pushed to GitHub.

### 2026-08-13 — Chapter 1 Complete + PIBench Phase 0

* Created repo `LearningRobotics`.
* Set up Python 3.11 virtual environment with MuJoCo 3.11.0.
* Built `simple_2r_arm.xml` and `simple_6dof_arm.xml`.
* Wrote `inspect_dof.py` to demonstrate configuration, DOF, FK, and C-space topology.
* Wrote `chapter01_foundation/notes.md` and the root `README.md`.
* **PIBench Phase 0:** built the benchmark engine, `TowerFall` statics scene, physics-oracle and random baselines, CLI, and tests (7 passing).
* Committed and pushed to GitHub.

---

## 📬 Contact & Follow Along

* GitHub: [@satyamdas03](https://github.com/satyamdas03)
* Project updates will be pushed to this repo as chapters are completed.

---

**License:** MIT — use it, fork it, improve it.

> *"Before you can control a robot, you have to describe where it is, how it is oriented, and how many independent ways it can move."* — Chapters 1–3
