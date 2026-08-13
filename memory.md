# 🧠 Project Memory — LearningRobotics

> **Purpose:** This file is the single source of truth for session restart. Read it first if you have no other context. It records everything we have done, decided, researched, and planned — line by line — so work can resume without loss.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project name** | LearningRobotics |
| **GitHub repository** | https://github.com/satyamdas03/LearningRobotics |
| **Branch** | `master` |
| **Visibility** | Public |
| **Mission** | Learn robotics and AI from first principles fast, and build something extraordinary and revolutionary that solves real-world problems. |
| **Owner** | Satyam Das (@satyamdas03, satyamdas03@gmail.com) |
| **Start date** | 2026-08-13 |
| **Current date** | 2026-08-13 |

---

## 2. Hardware & Environment

| Component | Spec |
|---|---|
| Laptop | Lenovo LOQ |
| GPU | NVIDIA GeForce RTX 5060 (8 GB VRAM) |
| NVIDIA driver | 595.95 |
| CUDA Version | 13.2 |
| OS | Windows 11 Home |
| Python version used | 3.11 (chosen because Isaac Sim/MuJoCo ecosystem is not yet stable on Python 3.14) |
| Primary shell | PowerShell; Bash available for POSIX commands |
| Physics simulator (current) | MuJoCo 3.11.0 |

**Why MuJoCo instead of Isaac Sim (for now):**
* Installs instantly via `pip install mujoco`.
* Runs cleanly on RTX 5060 without multi-GB downloads.
* Best-in-class contact physics — critical for manipulation.
* Native differentiability via MuJoCo XLA / MJX for future sim-to-real work.
* Isaac Sim / Isaac Lab will be added later for massively parallel RL and photoreal scenes.

---

## 3. Repository Layout

```
LearningRobotics/
├── README.md                            # Public-facing project overview + Chapter 1 log
├── memory.md                            # This file — full internal context for restarts
├── .gitignore                           # Ignore .venv/, __pycache__/, *.pyc, .DS_Store
└── chapter01_foundation/                  # Chapter 1 deliverables
    ├── requirements.txt                   # mujoco>=3.11.0, numpy
    ├── .venv/                             # Local Python 3.11 virtual environment (ignored by git)
    ├── simple_2r_arm.xml                  # 2-revolute planar arm (2 DOF)
    ├── simple_6dof_arm.xml                # 6-revolute spatial arm (6 DOF)
    ├── inspect_dof.py                     # DOF counter + FK demo + C-space topology demo
    └── notes.md                           # Chapter 1 session notes with observed numbers

Planned future structure (not yet created):
├── chapter02_rigid_body_motions/
├── chapter03_forward_kinematics/
├── ...
├── REVOLUTIONARY_ROBOTICS_IDEAS.md        # Already exists at root; see Section 7
```

---

## 4. Session History

### Session 1 — 2026-08-13

#### 4.1 What the user asked

User shared Chapter 1 of their robotics self-study curriculum (Foundations, Configuration Space, Degrees of Freedom) and asked to complete the practical exercise: install a simulator, load a robot, identify joints, count DOF, and observe joint-space vs task-space.

#### 4.2 What we built

1. **Created project directory** `C:\Users\point\projects\LearningRobotics\chapter01_foundation`.
2. **Created Python 3.11 virtual environment** `.venv` inside `chapter01_foundation`.
3. **Installed MuJoCo 3.11.0** plus dependencies (`numpy`, `glfw`, `pyopengl`, `absl-py`, `etils`).
4. **Created `requirements.txt`** pinning `mujoco>=3.11.0` and `numpy`.
5. **Created robot model `simple_2r_arm.xml`**:
   * Two revolute joints rotating about Z-axis.
   * Link lengths: L1 = 1.0 m, L2 = 0.8 m.
   * Represents a 2-DOF planar arm.
6. **Created robot model `simple_6dof_arm.xml`**:
   * Six revolute joints in 3D.
   * Joint axes: Z, Y, Y, X, Y, Z (waist, shoulder, elbow, wrist roll, wrist pitch, wrist yaw).
   * Represents a 6-DOF spatial arm matching the 6 DOF of a free rigid body.
7. **Created `inspect_dof.py`**:
   * Loads both robots.
   * Prints joint names, types, axes, qpos indices, ranges.
   * Prints `nq` (configuration-space dimension) and `nv` (velocity DOF).
   * Demonstrates forward kinematics: joint-space (qpos) → task-space (end-effector position).
   * Compares MuJoCo FK with analytical FK for the 2R arm.
   * Demonstrates C-space topology: revolute angles wrap around (0 = 2π), so 1 joint → circle, 2 joints → torus.
8. **Created `chapter01_foundation/notes.md`** with observed numbers and rerun instructions.
9. **Created root `README.md`** with full project overview, quick-start, and roadmap.
10. **Created `.gitignore`**.
11. **Initialized a new independent Git repository** in `C:\Users\point\projects\LearningRobotics` (separate from the giant `C:\Users\point` repo).
12. **Created GitHub repository** `satyamdas03/LearningRobotics` (public).
13. **Pushed root commit** `7439a5d` with all Chapter 1 files.

#### 4.3 Observed results from `inspect_dof.py`

**2R planar arm:**
* Joints: 2 revolute
* `nq` = 2, `nv` = 2
* Default end-effector: (1.800, 0.000, 0.050)
* FK verified against analytical solution for configurations (0,0), (π/4,0), (0,π/2), (0,π).

**6-DOF spatial arm:**
* Joints: 6 revolute
* `nq` = 6, `nv` = 6
* Default end-effector: (0.960, 0.000, 0.850)
* Configurations tested: all zeros; waist 90°; shoulder raised 45°; wrist pitch 45°.

**C-space topology:**
* q1=0 and q1=2π produced identical end-effector positions.
* Confirmed revolute-joint C-space is a circle (S¹); two revolute joints give a torus (S¹ × S¹).

#### 4.4 Rerun command

```powershell
cd C:\Users\point\projects\LearningRobotics\chapter01_foundation
. .venv\Scripts\Activate.ps1
python inspect_dof.py
```

---

### Session 2 — 2026-08-13 (continued)

#### 4.5 What the user asked

User asked:
1. Whether NVIDIA Isaac Sim would run on their laptop.
2. Whether MuJoCo is good enough long-term for revolutionary work.
3. To expand on "new algorithms, new representations, new control structures, new AI + physics integration" with deep research, analysis, and brainstorming of genuinely new, extraordinary, revolutionary ideas that solve real-world blockers.

#### 4.6 Isaac Sim feasibility assessment

* **Verdict:** RTX 5060 with 8 GB VRAM can run Isaac Sim / Isaac Lab for learning, but not at production scale.
* **Recommended approach:** Use `pip install isaacsim` (lighter, Python-first, headless) rather than full Omniverse launcher.
* **Limitations on this hardware:** 8 GB VRAM is the bottleneck; large scenes and thousands of parallel RL envs will crash or crawl.
* **Practical rules:** run headless, start with simple robots, limit parallel envs to 64 or fewer, disable RTX eye candy, monitor VRAM.
* **Cloud alternative:** RunPod / Vast.ai / Lambda for large-scale training bursts.
* **Timeline:** Stick with MuJoCo for Chapters 1–5; add Isaac Sim/Lab when reaching RL chapters.

#### 4.7 MuJoCo long-term verdict

* **Verdict:** MuJoCo is more than good enough for revolutionary robotics research.
* **Evidence:** It is the engine behind much of OpenAI, DeepMind, Google, MIT, Stanford robotics work; OpenAI Gym was built on it; best-in-class contact physics; native differentiability via MJX.
* **Weaknesses vs Isaac:** No massive GPU parallelization out-of-the-box (but MJX exists); no photoreal rendering pipeline; smaller curated asset ecosystem.
* **Recommended hybrid strategy:** Use MuJoCo for fast prototyping, contact-rich control, and algorithmic research; use Isaac Sim/Lab for GPU-parallel RL and photoreal digital twins; use real robot as soon as possible.
* **Why it fits this project:** The revolutionary part is architecture/insight, not the renderer. MuJoCo is ideal for the top-ranked concept ideas.

#### 4.8 Deep-research / brainstorming process

Ran parallel web searches across:
* Foundation models / VLA for manipulation (2025–2026)
* Neuro-symbolic task and motion planning (2025–2026)
* Causal RL / world models for robotics (2025–2026)
* Contact-rich manipulation / dexterous robots (2025–2026)
* Differentiable physics / sim-to-real (2025–2026)
* Embodied AI / LLMs / physics reasoning (2025–2026)
* Unsolved problems in robotics (2025–2026)
* Low-cost robot arms under $1000 (2025–2026)
* Elderly-care robotics blockers (2025–2026)
* Construction robotics / labor shortage (2025–2026)
* Open-source robot hardware platforms (2025–2026)

#### 4.9 Key state-of-the-art snapshots identified

**VLA / Foundation Models for Robotics:**
* [DexSim2Real](https://arxiv.org/html/2605.05241) — foundation-model-guided sim-to-real, 78.2% real-world success.
* [ABot-M0](https://arxiv.org/html/2602.11236) — VLA on 6M+ trajectories, Action Manifold Learning.
* [Gemini Robotics](https://doi.org/10.48550/arxiv.2503.20020) — end-to-end VLA at ~50 Hz.
* [Vesta](https://doi.org/10.48550/arxiv.2606.20905) — NVIDIA generalist embodied reasoning model.
* [Embodied-R1.5](https://arxiv.org/html/2606.11324) — 8B embodied foundation model.
* [Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement](https://www.microsoft.com/en-us/research/articles/object-centric-residual-rl/) — residual policy lifts real-robot success 42% → 76%.

**Physics-Grounded AI:**
* [PhysVLA](https://arxiv.org/html/2606.13886) — inference-time physics gate around frozen VLA.
* [SIMPACT](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_SIMPACT_Simulation-Enabled_Action_Planning_using_Vision-Language_Models_CVPR_2026_paper.html) — test-time simulation-in-the-loop planning.
* [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998v2) — autoregressive diffusion world model.
* [DexWorldModel](https://arxiv.org/html/2604.16484) — causal latent world model with DINOv3.
* [World4RL](https://arxiv.org/html/2509.19080v2) — diffusion world models for policy refinement.

**Differentiable Simulation / Sim-to-Real:**
* [HALO](https://arxiv.org/abs/2603.15084v1) — MJX + analytical gradients for heavy-loaded humanoid system ID, zero-shot transfer.
* [DiffMJX](https://arxiv.org/html/2506.14186v1) — correct contact gradients in penalty-based sim.
* [RSR Loop](https://arxiv.org/html/2503.10118v2) — iterative real-sim-real tuning with InfoGap loss.
* [D-REX](https://arxiv.org/html/2603.01151) — Gaussian Splatting + differentiable physics for dexterous grasping.
* [Learning Deployable Locomotion via Differentiable Simulation](https://proceedings.mlr.press/v305/schwarke25a.html) — zero-shot sim-to-real from differentiable sim.

**Contact-Rich Manipulation:**
* [Environmental Constraints](https://arxiv.org/html/2605.17601) — generalize from one demonstration.
* [FA-RDP](https://arxiv.org/html/2607.28596) — frequency-adaptive reactive diffusion policy.
* [PhaForce](https://arxiv.org/html/2603.08342v1) — slow-fast visual-force policy with contact phases.
* [Data and Learning Where it Matters](https://arxiv.org/html/2607.15982) — dense data only for contact segment + classical planning elsewhere.
* [Tube Diffusion Policy](https://arxiv.org/html/2604.23609v1) — diffusion nominal + feedback flow.

**Neuro-Symbolic Planning:**
* [H-WM](https://arxiv.org/abs/2602.11291v2) — hierarchical world model for TAMP.
* [Flax](https://doi.org/10.1109/lra.2026.3662556) / [LLM-Flax](https://arxiv.org/html/2604.26569) / [iFlax](https://arxiv.org/html/2606.06877) — neuro-symbolic relaxation planners.
* [Learning Sound Symbolic Abstractions from VLMs](https://openreview.net/forum?id=liVUIlgUI5) — VLM → PDDL predicates.

**Unsolved Real-World Blockers Identified:**
1. Real robot data is expensive ($4–$30 per episode).
2. Sim-to-real gap for contact-rich tasks.
3. Long-horizon error compounding (10-step tasks work; 100-step tasks fail).
4. Dexterous in-hand manipulation.
5. Physical property understanding (mass, deformability, fragility).
6. Personalization and lifelong learning.
7. Cost and accessibility of hardware.
8. Safety without conservatism.
9. Evaluation fragmentation.
10. Unstructured real homes (clutter, pets, stairs, lighting, connectivity).

**Low-Cost Open Hardware Identified:**
* [AM-ARM ~$380](https://github.com/liyiteng/AM-ARM) — 6+1 DoF arm, 1 kg payload, 52 cm reach.
* [Forte ~$215](https://arxiv.org/html/2507.15693) — 6-DoF arm, sub-mm repeatability.
* [Low-Cost Robot Arm](https://github.com/AlexanderKoch-Koch/low_cost_robot) — ~$430 leader+follower.
* [Open Arms Mini ~€150](https://github.com/pkooij/open-arms-mini) — 7-DoF leader arm.
* [U-ARM ~$50](https://arxiv.org/html/2509.02437) — ultra-low-cost teleop leader.
* [Zeroth-01 Bot](https://github.com/zeroth-robotics/zeroth-bot) — open-source humanoid, ~$350.
* [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm/) — 6-DOF + gripper, ROS/Pinocchio/LeRobot/Isaac Sim stack.

**Impact Domains Identified:**
* Elderly care at home — need reliable, personalized assistance in unstructured homes.
* Construction — labor shortage, need autonomous robust contact-aware work.
* Warehouse logistics — generalize to unseen objects, handle deformable/fragile items.
* Industrial assembly — precision insertion, long-horizon tasks, fast reprogramming.
* Education / hobby robotics — cheap hardware + accessible programming.

#### 4.10 Revolutionary concept proposals created

Wrote `REVOLUTIONARY_ROBOTICS_IDEAS.md` (pushed to GitHub as commit `4195d24`). Contains 12 concrete concepts plus 2 enablers:

1. **Universal Skill Compiler** — video → physics reconstruction → sim → policy.
2. **Physics-Grounded Chain-of-Thought Robot** — LLM proposes, physics engine verifies, failures feedback to planner.
3. **Causal Manifold of Skills** — learn skill representations with causal dimensions.
4. **Self-Improving Real-Sim-Real Loop** — robot detects failures, updates sim, retrains, retries.
5. **Affordance-First Perception** — output field of physically possible actions.
6. **Constraint-Language Robot Programming** — natural language + sketch → formal constraints + safe controller.
7. **Embodied World Model from Internet Video** — physical-intuition world model trained on internet video.
8. **Dexterous Manipulation as Contact Grammar** — compositional grammar of contact primitives.
9. **Uncertainty-Aware Autonomous Robot** — calibrated uncertainty; asks for help when uncertain.
10. **100-Robot Skill-Sharing Network** — federated latent skill sharing across cheap robots.
11. **Wearable Teleoperation for Robot Teachers** — cheap glove + phone → robot demonstrations.
12. **Physical Intuition Benchmark** — standardized physical-reasoning benchmark.

#### 4.11 Top 5 ranked concepts for this project

| Rank | Concept | Why it wins | Fits current setup? |
|---|---|---|---|
| 1 | Universal Skill Compiler | Solves #1 blocker: data scarcity. | ✅ MuJoCo + VLMs |
| 2 | Physics-Grounded Chain-of-Thought | Makes VLMs trustworthy for physical tasks. | ✅ GPT-4o-mini + MuJoCo |
| 3 | Self-Improving Real-Sim-Real Loop | Autonomous sim-to-real gap closing. | ⚠️ Needs cheap real arm |
| 4 | Uncertainty-Aware Autonomous Robot | Trustworthy autonomy; wraps any policy. | ✅ Can add to any base policy |
| 5 | Physical Intuition Benchmark | Standardizes evaluation; entirely in MuJoCo. | ✅ Zero hardware cost |

#### 4.12 North star agreed

> **Build a robot that learns physical skills from ordinary video, reasons about them with a physics engine, executes them with provable safety, and shares what it learns with other robots — for less than $500 in hardware.**

#### 4.13 Recommended attack plan (phased)

**Phase 0: Foundations (now — Chapters 2–5)**
* Master rigid-body transforms, forward/inverse kinematics, Jacobians, dynamics.
* Build reusable MuJoCo models.
* Side project: start the **Physical Intuition Benchmark** (Concept L).

**Phase 1: Skill Representation (Chapters 6–7)**
* Library of contact primitives in MuJoCo (Concept H seed).
* Constraint-language programming for simple tasks (Concept F seed).
* Uncertainty-aware control wrappers (Concept I seed).

**Phase 2: Learning from Observation**
* Video → sim reconstruction for tabletop tasks (Concept A prototype).
* Train affordance maps from synthetic data (Concept E prototype).
* Optional: build wearable teleop glove (Concept K).

**Phase 3: Real-Sim-Real Loop**
* Purchase cheap arm (AM-ARM ~$380 or Forte ~$215).
* Implement self-improving calibration loop (Concept D).
* Publish real-world results, open-source everything.

**Phase 4: Foundation-Model Integration**
* LLM/VLM planning with physics verification (Concept B).
* Causal skill manifolds (Concept C).
* Connect to skill-sharing network idea (Concept J).

#### 4.14 What the user asked for at the end of Session 2

User asked for a `memory.md` file at the project root that stores all context end-to-end so that if the session restarts and memory is lost, the model can read it and regain full context. This file is the response to that request.

---

## 5. Current Status Snapshot

| Area | Status |
|---|---|
| Chapter 1 practical | ✅ Complete and pushed |
| GitHub repo | ✅ Live at https://github.com/satyamdas03/LearningRobotics |
| README | ✅ Complete |
| Revolutionary manifesto | ✅ Complete and pushed |
| Chapter 2 reading | 🚧 User currently reading Chapter 2 (Rigid-Body Motions) |
| Next implementation work | ⏳ Not started — decide based on user's next message |
| Hardware purchase | ⏳ None yet; consider AM-ARM / Forte / U-ARM in Phase 3 |
| Isaac Sim installed | ⏳ Not installed; will revisit for RL chapters |

---

## 6. Open Decisions / Questions

1. Should the next immediate project be the **Physical Intuition Benchmark** while continuing Chapter 2 study, or should we follow the textbook linearly first?
2. Which cheap robot arm should be the long-term hardware target? AM-ARM ($380, 6+1 DoF, 1 kg payload) is the leading candidate for capability; Forte ($215) is cheapest for a real manipulator; U-ARM ($50) is best for teleop-only data collection.
3. Should the README or manifesto be converted into a polished website / artifact for sharing?
4. Should we install Isaac Sim in headless pip mode now to verify it runs on the RTX 5060, or wait until needed?

---

## 7. Important File Index

| File | Purpose | When to read / update |
|---|---|---|
| `memory.md` | Full project context for restarts | **Read first on every restart** |
| `README.md` | Public project overview | Update after each chapter |
| `REVOLUTIONARY_ROBOTICS_IDEAS.md` | Research manifesto + 12 concepts | Update as ideas evolve |
| `chapter01_foundation/notes.md` | Chapter 1 session notes | Reference for Chapter 1 details |
| `chapter01_foundation/inspect_dof.py` | Chapter 1 runnable demo | Run whenever showing Chapter 1 |
| `chapter01_foundation/simple_2r_arm.xml` | 2-DOF robot model | Reuse/extend for kinematics chapters |
| `chapter01_foundation/simple_6dof_arm.xml` | 6-DOF robot model | Reuse/extend for kinematics/control chapters |
| `.gitignore` | Git exclusions | Update if new tooling adds artifacts |

---

## 8. Commands That Work

### Run Chapter 1 demo

```powershell
cd C:\Users\point\projects\LearningRobotics\chapter01_foundation
. .venv\Scripts\Activate.ps1
python inspect_dof.py
```

### Commit and push future changes

```powershell
cd C:\Users\point\projects\LearningRobotics
git add .
git commit -m "robotics: descriptive message here"
git push origin master
```

Note: this repo is independent of the `C:\Users\point` mega-repo. Do not accidentally stage files outside `LearningRobotics`.

---

## 9. Design Principles Established

1. **Build in public.** Repo is public; document everything.
2. **MuJoCo first.** Start with MuJoCo; add Isaac Sim only when a specific problem demands it.
3. **Cheap hardware later.** Use simulation for as long as possible; buy real hardware only when an idea is validated in sim.
4. **Lineage over silos.** Every experiment should be reproducible, committed, and linked to a concept from the manifesto.
5. **Solve real blockers.** Revolutionary ideas target unsolved problems, not incremental improvements.
6. **Physics-aware AI.** Any AI planner must be grounded in a physics engine; language is not enough.

---

## 10. Summary for Fast Restart

If you are resuming this session with no other context, here is the one-paragraph summary:

> We are building `LearningRobotics`, a public learning journal and research lab for robotics + AI. Chapter 1 is complete in MuJoCo with a 2R arm and a 6-DOF arm, plus a DOF inspector. We researched the 2025–2026 frontier and wrote a manifesto with 12 revolutionary project ideas targeting real-world blockers like data scarcity, sim-to-real, long-horizon planning, and cheap hardware. The top immediate candidate is the **Physical Intuition Benchmark** — a standardized MuJoCo benchmark for physical common-sense reasoning that can be built during Chapters 2–5. The long-term north star is a sub-$500 robot that learns from video, reasons with physics, executes safely, and shares skills with other robots.

---

*Last updated: 2026-08-13*
