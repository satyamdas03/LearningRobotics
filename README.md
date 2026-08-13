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
| **P0** | **PIBench — Physical Intuition Benchmark** | ✅ Phase 0 Complete | `pibench/` engine + `TowerFall` statics scene, tests passing |
| 2 | Rigid-Body Motions (frames, rotations, transforms) | 🚧 Next | TBD |
| 3 | Forward Kinematics | ⏳ Planned | TBD |
| 4 | Velocity Kinematics & Jacobians | ⏳ Planned | TBD |
| 5 | Inverse Kinematics | ⏳ Planned | TBD |
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
└── pibench/                       # Physical Intuition Benchmark (Phase 0)
    ├── README.md                  # PIBench overview and quickstart
    ├── requirements.txt           # MuJoCo + benchmark deps
    ├── pibench/                   # Python package
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

### 3. Run PIBench

```powershell
cd pibench
. .venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Users\point\projects\LearningRobotics\pibench"
python -m pibench run --suite statics --predictor physics_oracle --n 5
```

Expected output: `Overall accuracy: 100.0%` on the `TowerFall` scene.

---

## 🧠 Test Yourself — Chapter 1 Answers

1. **Drone flying freely** — **6 DOF**: `(x, y, z)` position + roll, pitch, yaw.
2. **Smartphone sliding/spinning on a table** — **3 DOF**: `x, y` position + rotation θ about the table normal.
3. **5 revolute joints in an open chain** — **5 DOF**. It cannot reach every position *and* orientation in 3D because it has fewer DOF than the 6 DOF of a free rigid body.
4. **Why C-space reframing helps planning** — It turns a complex physical robot into a single point moving through an abstract space. Obstacles become forbidden regions, and planning becomes "find a path for the point."
5. **Planar mechanism with 5 links and 5 revolute joints** — Using Grübler's formula with `m=3`, `N=5`, `J=5`, `Σf_i=5`:
   `DOF = 3(5 - 1 - 5) + 5 = 3(-1) + 5 = 2`.

---

## 🧱 PIBench — First Revolutionary Deliverable (Phase 0)

PIBench is the executable first step toward the *Revolutionary Robotics* north star. It evaluates whether a model truly understands physical common sense, starting with static stability.

### What works now

* **Engine scaffold:** `Problem`, `Suite`, `Runner`, `Evaluator`, `Registry`, and model-agnostic `Predictor` interface.
* **First scene:** `TowerFall` — predict which of two towers falls when the platform tilts.
* **Baselines:** `physics_oracle` (100% on deterministic scenes) and `random`.
* **CLI:** `pibench list`, `pibench run`, `pibench render`.
* **Tests:** 7 passing.

### Run it

```powershell
cd pibench
. .venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Users\point\projects\LearningRobotics\pibench"
python -m pibench run --suite statics --predictor physics_oracle --n 10
```

Expected output: `Overall accuracy: 100.0%` on the `TowerFall` scene.

### Why this matters

A model that can answer "which tower falls?" is being forced to reason about center of mass and support polygon — the same concepts in Chapter 2. As the textbook progresses, PIBench progresses: dynamics, contact, articulated bodies, and finally parameter estimation / counterfactuals.

---

## 🗺️ Roadmap to "Extraordinary & Revolutionary"

The end goal is not just to understand robotics, but to build an AI-driven robotic system that does something genuinely novel. Possible north stars:

* A **self-improving manipulator** that learns from few demonstrations using RL + world models.
* A **skill library** stored as reusable motion primitives in C-space.
* A **sim-to-real pipeline** that transfers policies from MuJoCo/Isaac Sim to an affordable real robot arm.
* A **foundation-model robot brain** that interprets natural-language tasks and synthesizes motion plans + control policies.

Each chapter feeds into that stack. Chapter 1 is the foundation: know the links, joints, DOF, and C-space.

---

## 📝 Changelog

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

> *"Before you can control a robot, you have to describe where it is and how many independent ways it can move."* — Chapter 1
