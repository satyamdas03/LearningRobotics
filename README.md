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
| 6 | Dynamics | ⏳ Planned | TBD |
| 7 | Control | ⏳ Planned | TBD |
| 8 | Motion Planning | ⏳ Planned | TBD |
| 9 | Reinforcement Learning with Isaac Lab | ⏳ Planned | TBD |
| 10 | Real-world embodiment / integration | ⏳ Planned | TBD |

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
└── pibench/                       # Physical Intuition Benchmark (Phases 0-4)
    ├── README.md                  # PIBench overview and quickstart
    ├── requirements.txt           # MuJoCo + benchmark deps
    ├── pibench/                   # Python package
    │   ├── scenes/statics/        # TowerFall, SlopeSlide, SupportBalance, ToppleDirection
    │   ├── scenes/dynamics/       # PendulumSwing, CollisionBounce, ProjectileHit
    │   ├── scenes/contact/        # PushTipVsSlide, StackStability, WedgeInsert, FrictionPile, SlipGrip
    │   ├── scenes/articulated/    # DrawerPull, DoorSwing, RopeTension, GearTurn
    │   ├── scenes/deformable/     # ChainDrape
    │   └── utils/                 # MJCF + contact + articulated helpers
    ├── tests/                     # pytest suite
    └── run_all.py                 # Run all suites across all predictors
```

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

### 6. Run PIBench

```powershell
cd pibench
. .venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Users\point\projects\LearningRobotics\pibench"
python -m pibench run --suite statics --predictor physics_oracle --n 5
python -m pibench run --suite dynamics --predictor physics_oracle --n 5
python -m pibench run --suite contact --predictor physics_oracle --n 5
python -m pibench run --suite articulated --predictor physics_oracle --n 5
python -m pibench run --suite deformable --predictor physics_oracle --n 5
python run_all.py
```

Expected output: `Overall accuracy: 100.0%` on every suite. Physics-oracle should be the top predictor.

---

## 🧠 Test Yourself — Chapter 1 Answers

1. **Drone flying freely** — **6 DOF**: `(x, y, z)` position + roll, pitch, yaw.
2. **Smartphone sliding/spinning on a table** — **3 DOF**: `x, y` position + rotation θ about the table normal.
3. **5 revolute joints in an open chain** — **5 DOF**. It cannot reach every position *and* orientation in 3D because it has fewer DOF than the 6 DOF of a free rigid body.
4. **Why C-space reframing helps planning** — It turns a complex physical robot into a single point moving through an abstract space. Obstacles become forbidden regions, and planning becomes "find a path for the point."
5. **Planar mechanism with 5 links and 5 revolute joints** — Using Grübler's formula with `m=3`, `N=5`, `J=5`, `Σf_i=5`:
   `DOF = 3(5 - 1 - 5) + 5 = 3(-1) + 5 = 2`.

---

## 🧱 PIBench — First Revolutionary Deliverable (Phases 0–4)

PIBench is the executable first step toward the *Revolutionary Robotics* north star. It evaluates whether a model truly understands physical common sense across statics, dynamics, contact/friction, articulated constraints, and deformable bodies.

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
* **Baselines:** `physics_oracle` (100% on deterministic scenes), `random`.
* **CLI:** `pibench list`, `pibench run`, `pibench render`, `pibench view`.
* **Tests:** 44 passing across engine + all scenes.

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
python -m pibench view TowerFall
python run_all.py
```

Expected output: `Overall accuracy: 100.0%` on every suite. `physics_oracle` tops the leaderboard.

### Why this matters

A model that can answer "which tower falls?" or "where does the projectile land?" is being forced to reason about center of mass, support polygon, friction, energy, and projectile motion — the same concepts in Chapters 2 and 3. With the contact suite it must also reason about moment balance, impulse, contact geometry, and Coulomb friction — the concepts in Chapter 4. With the articulated/deformable suite it must reason about constraints, prismatic and revolute joints, tendons, and coarse deformable approximations — the concepts in Chapter 5.

---

## 🗺️ Roadmap to "Extraordinary & Revolutionary"

The end goal is not just to understand robotics, but to build an AI-driven robotic system that does something genuinely novel. Possible north stars:

* A **self-improving manipulator** that learns from few demonstrations using RL + world models.
* A **skill library** stored as reusable motion primitives in C-space.
* A **sim-to-real pipeline** that transfers policies from MuJoCo/Isaac Sim to an affordable real robot arm.
* A **foundation-model robot brain** that interprets natural-language tasks and synthesizes motion plans + control policies.

Each chapter feeds into that stack. Chapter 1 is the foundation (links, joints, DOF, C-space). Chapters 2 and 3 add the math of rigid-body motion and the geometry of forward kinematics needed for any real manipulator.

---

## 📝 Changelog

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
