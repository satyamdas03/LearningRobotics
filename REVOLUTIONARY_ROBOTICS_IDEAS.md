# 🚀 Revolutionary Robotics Research Manifesto

> **"Don't build a slightly better robot. Build a robot capability that changes what is possible."**

This document is the output of a deep-dive brainstorming session on what is genuinely new, extraordinary, and revolutionary in robotics/AI — and what could actually be built with the tools we already have (MuJoCo, Python, an RTX 5060) plus ambition.

---

## 1. The Lens: What Counts as "Revolutionary"?

An idea is revolutionary if it does **one or more** of the following:

1. **Solves a problem that currently has no good solution** (not just 10% better).
2. **Changes the economics by 10×** — cheaper data, cheaper hardware, cheaper deployment.
3. **Unlocks a capability that does not exist yet** in deployed systems.
4. **Combines fields in a way no one has done** — e.g., causal inference + control + LLMs.
5. **Has a clear path to real-world impact** — not just a paper metric.

The goal is not to compete with NVIDIA or Google on scale. The goal is to out-think them on **architecture** and **insight**.

---

## 2. State of the Art Snapshots (2025–2026)

### 2.1 Vision-Language-Action (VLA) Models

The dominant bet for generalist robot control is to train giant models on internet-scale vision-language data plus robot trajectories:

* **ABot-M0** — 6M+ trajectories, 9,500+ hours, Action Manifold Learning; SOTA on LIBERO / RoboTwin.
* **Gemini Robotics** — DeepMind's end-to-end VLA at ~50 Hz, handles deformable objects.
* **Vesta** — NVIDIA's generalist embodied reasoning model unifying localization, navigation, QA, planning.
* **Embodied-R1.5** — 8B VLM that turns into a VLA with small action-data fine-tuning.

**The catch:** These are data-hungry, black-box, and can hallucinate physical plans. They often fail on contact-rich or long-horizon tasks.

### 2.2 Physics-Grounded AI

Researchers are trying to make foundation models respect physics:

* **PhysVLA** — wraps a frozen VLA with a phase-aware finite-state machine + Euler-Lagrange gate; 35% → 95% success on real pick-and-place.
* **SIMPACT** — test-time simulation-in-the-loop: VLM proposes actions, simulator rolls them out, plan refines.
* **Causal World Modeling for Robot Control** — diffusion world models with causal attention.
* **DexWorldModel** — causal latent world model using DINOv3 features for sim-to-real transfer.

**The catch:** Most are inference-time wrappers or training corrections. They don't deeply integrate physics into learning.

### 2.3 Differentiable Simulation & Sim-to-Real

This is the most practical frontier for a single developer:

* **HALO** — MuJoCo MJX + analytical gradients for system ID; zero-shot humanoid transfer.
* **DiffMJX** — fixes contact gradients in MuJoCo XLA for real-world system identification.
* **RSR Loop** — Real-Sim-Real iterative tuning with InfoGap loss.
* **D-REX** — Gaussian Splatting + differentiable physics for dexterous grasping.

**The catch:** These focus on *parameters* (mass, friction, CoM). They don't yet close the gap on *unmodeled phenomena* like deformable objects, wear, or human contact.

### 2.4 Contact-Rich Manipulation

The hardest frontier in manipulation:

* **Environmental Constraints** — abstract one demonstration into physical constraints for generalization.
* **FA-RDP** — frequency-adaptive diffusion policy: slow before contact, fast after.
* **PhaForce** — slow planner + fast corrector with explicit contact-phase schedule.
* **Tube Diffusion Policy** — diffusion nominal trajectory + feedback flow for disturbance rejection.

**The catch:** Most require real robot data or task-specific engineering.

### 2.5 Neuro-Symbolic Planning

Combining neural perception with symbolic planners:

* **H-WM** — hierarchical world model for TAMP.
* **Flax / LLM-Flax / iFlax** — GNN/LLM object importance + symbolic relaxation.
* **Learning Sound Symbolic Abstractions from VLMs** — convert VLM confidences into PDDL predicates.

**The catch:** Symbolic abstractions are hand-engineered or brittle. They don't learn from failure.

---

## 3. The Real Unsolved Blockers

These are the gaps that currently prevent robots from leaving the lab:

| Blocker | Why it matters | Who suffers |
|---|---|---|
| **1. Real robot data is expensive** | $4–$30 per episode; generalist policies need 10k+ episodes | Academics, startups, hobbyists |
| **2. Sim-to-real gap for contact** | Free-space transfer works; contact-rich tasks fail | Industrial assembly, household robots |
| **3. Long-horizon error compounding** | 10-step tasks work; 100-step tasks fall apart | Any real-world task |
| **4. Dexterous in-hand manipulation** | Multi-finger reorientation, tool use, in-hand rotation | Logistics, manufacturing, home care |
| **5. Physical property understanding** | Robots don't know mass, deformability, fragility | Handling food, cloth, glass, tools |
| **6. Personalization & lifelong learning** | One-size-fits-all; retraining is expensive | Elderly care, homes, small businesses |
| **7. Cost & accessibility** | Useful arms cost $20k+; hardware is the barrier | Developers, researchers, developing world |
| **8. Safety without conservatism** | Safe policies are slow; fast policies are unsafe | Collaborative manufacturing, home care |
| **9. Evaluation fragmentation** | No common benchmarks; results are incomparable | Whole field |
| **10. Unstructured real homes** | Clutter, pets, stairs, poor lighting, unreliable internet | Home robotics |

The **central bottleneck** is **transfer and generalization**: handling new objects, environments, and tasks without expensive retraining.

---

## 4. Revolutionary Concept Proposals

These are concrete architectural ideas. Each addresses at least one blocker and could plausibly be built incrementally.

---

### Concept A: The Universal Skill Compiler

**The problem:** Teaching a robot a new physical skill today requires either expensive teleoperation data or expert engineering. A human can watch a YouTube video and learn; a robot cannot.

**The idea:** A pipeline that takes a single video of a human performing a manipulation task and outputs:
1. A **physics-grounded scene reconstruction** (object shapes, initial/final poses, contacts).
2. A **constraint abstraction** — the physical invariants of the task (e.g., "cup must stay upright," "plug must align with socket axis").
3. A **simulation-ready task specification** in MuJoCo.
4. A **deployable policy** trained in sim and transferred to a cheap real arm.

**Why revolutionary:** Removes the data-collection bottleneck. Anyone can teach a robot by showing it a video.

**Architecture:**
```
YouTube video
    ↓
VLM / video foundation model → parse task, detect objects, track hands
    ↓
Neural reconstruction (Gaussian Splatting / DUSt3R) → 3D scene + object meshes
    ↓
Physics engine (MuJoCo) → fit object dynamics, identify contact constraints
    ↓
Constraint abstraction module → convert to skill grammar
    ↓
Imitation / RL in sim → train policy
    ↓
Deploy on AM-ARM / Forte / real robot
```

**Blockers solved:** Data scarcity, sim-to-real setup cost, accessibility.

**Feasibility on your setup:** High. MuJoCo for simulation; off-the-shelf VLMs for parsing; no massive GPU needed if you start with simple objects.

---

### Concept B: Physics-Grounded Chain-of-Thought Robot

**The problem:** LLM/VLAs plan in language and vision, not physics. They propose actions that violate dynamics, miss contacts, or ignore stability.

**The idea:** A robot reasoning loop where every proposed high-level action is **simulated before execution**. If the simulation succeeds, the action proceeds. If it fails, the LLM receives a structured failure report and revises the plan.

```
Instruction: "Place the cup on the edge of the table."
    ↓
LLM proposes plan: [grasp cup, move over table, release]
    ↓
MuJoCo rollout (with learned residual dynamics)
    ↓
Simulation result: cup falls off table → FAIL
    ↓
LLM revises: [grasp cup, move to center of table, release]
    ↓
Simulation result: stable → EXECUTE
```

**Why revolutionary:** Makes foundation models **trustworthy for physical tasks**. It turns a language model into a physics-aware planner, not a pattern matcher.

**Architecture:**
* **Planner:** LLM / VLM generating symbolic action sequences.
* **Physics oracle:** MuJoCo + learned residual dynamics.
* **Verifier:** Checks stability, contact feasibility, collision, task success.
* **Feedback loop:** Structured error messages back to planner.
* **Controller:** Low-level MPC or diffusion policy executes verified actions.

**Blockers solved:** Physical world understanding, hallucination in VLA planning, safety.

**Feasibility:** Medium-High. You don't need a giant model. Start with GPT-4o-mini + MuJoCo + a simple tabletop scene.

---

### Concept C: Causal Manifold of Skills

**The problem:** Robots memorize trajectories, not skills. A policy trained to pick red cups fails on blue cups of different shapes. There is no representation of *what makes this skill work*.

**The idea:** Learn a low-dimensional "skill manifold" where each dimension corresponds to a **causal variable** of the task:
* Grasp aperture vs object diameter.
* Approach angle vs object center of mass.
* Insertion depth vs socket depth.
* Force threshold vs surface friction.

The robot can then **interpolate** to new objects and **extrapolate** by adjusting causal knobs.

**Why revolutionary:** True generalization in manipulation. Instead of memorizing, the robot understands the causal structure of the skill.

**Architecture:**
* Collect multi-object demonstrations.
* Train an object-centric encoder + latent skill model.
* Use **causal discovery** (e.g., NOTEARS, ICA) or intervention experiments to identify causal dimensions.
* Train a **causal policy** that conditions on inferred causal variables.
* Deploy with online causal inference at test time.

**Blockers solved:** Generalization to novel objects, data efficiency, interpretability.

**Feasibility:** Medium. Requires careful experimental design for causal identification.

---

### Concept D: Self-Improving Virtual Real-Sim-Real Loop

**The problem:** Sim-to-real is usually a one-way street: build sim, train, deploy, hope it works. When it fails, you don't know why. We solve it first in simulation by building a virtual arm whose internal physics model can differ from the controller's assumed model, then closing the loop to identify and correct that mismatch.

**The idea:** A closed loop where the real robot identifies its own failures, updates the simulator's parameters *and structure*, retrains, and tries again — autonomously.

**Why revolutionary:** The robot closes its own sim-to-real gap. It becomes a self-calibrating system.

**Architecture:**
1. Real robot attempts task.
2. **Failure detector** compares expected vs observed trajectory.
3. **Differentiable simulator** identifies which parameters (mass, friction, damping, contact stiffness) best explain the mismatch.
4. **Structure learner** asks: is the mismatch due to a missing physics phenomenon (deformation, slip, backlash)? If so, add a new model component.
5. **Retrain policy** in updated sim.
6. **Retry** with informative exploration.

**Blockers solved:** Sim-to-real gap, lifelong learning, deployment cost.

**Feasibility:** Medium. Requires a real robot (even a cheap one) and a differentiable sim stack (MuJoCo MJX + DiffMJX).

---

### Concept E: Affordance-First Perception

**The problem:** Robots detect *objects*; they don't detect *possibilities*. A human sees a chair and knows they can sit on it, push it, stand on it, or drag it. A robot sees a point cloud.

**The idea:** Train a perception system that outputs an **affordance field**: for every point in the scene, predict what actions are physically possible and what their outcomes would be.

```
Input: RGB-D image of a kitchen
Output:
  - handle: grasp + pull → drawer opens
  - cup: grasp + tilt → liquid pours
  - table: stable support region
  - cloth: pinch + drag → moves
```

**Why revolutionary:** Radically reduces exploration. The robot already knows what it can do before it acts.

**Architecture:**
* Foundation vision model for segmentation + 3D understanding.
* Physics-prior network predicts stability, graspability, motion constraints.
* Learned affordance map from passive video + physical common sense.
* Integrated with planner for action proposal.

**Blockers solved:** Unstructured environments, long-horizon planning, sample efficiency.

**Feasibility:** High. Can be trained from internet video + sim. Start with tabletop objects.

---

### Concept F: Constraint-Language Robot Programming

**The problem:** Programming a robot today requires ROS, trajectory planners, and control expertise. Natural language instructions like "place the cup upright without spilling" are not executable.

**The idea:** A system that converts natural language + a rough sketch into **formal constraints + a learned controller** with safety guarantees.

Example input: "Pour water from the bottle into the cup without spilling and keep the cup upright."

Output:
* Hard constraints: cup upright angle < 10°, bottle above cup mouth during pour.
* Soft preferences: minimize splash, minimize time.
* Controller: CBF-QP + learned residual policy.

**Why revolutionary:** Makes robot programming accessible to non-experts with **provable safety**.

**Architecture:**
* LLM parses instruction into formal constraint language (e.g., linear temporal logic + bounds).
* Symbolic verifier checks constraint consistency and feasibility.
* Control barrier functions (CBFs) enforce hard constraints in real time.
* Learned policy optimizes within the safe set.

**Blockers solved:** Accessibility, safety, human-robot collaboration.

**Feasibility:** Medium. Start with simple constraints (stay upright, avoid collision) and scale.

---

### Concept G: Embodied World Model from Internet Video

**The problem:** World models are usually trained on robot data. There isn't enough robot data. But there is infinite video of humans interacting with the physical world.

**The idea:** Train a world model on massive internet video to predict *physical outcomes*:
* "If I push this cup, it slides and may fall."
* "If I pull this drawer, it opens."
* "If I drop this egg, it breaks."

Then fine-tune a small action head for robot control.

**Why revolutionary:** Internet-scale physical knowledge without a single robot demonstration.

**Architecture:**
* Video foundation model (pre-trained on internet video).
* Physics head predicts future states as object-centric tokens + contact events.
* Contrastive/objective: predict outcomes of physical interventions.
* Transfer to robot: replace human hand with robot end-effector in latent space.

**Blockers solved:** Data scarcity, physical property understanding, generalization.

**Feasibility:** Low-Medium for an individual. Requires large-scale video data and compute. But can be scoped down to a narrow domain (e.g., tabletop physics) on an RTX 5060.

---

### Concept H: Dexterous Manipulation as Contact Grammar

**The problem:** Dexterous manipulation is treated as end-to-end motion. There is no compositional structure — no "vocabulary" of contacts.

**The idea:** Define a **formal grammar of contact primitives**:
* **Approach** → **Touch** → **Slide** → **Grasp** → **Lift** → **Reorient** → **Place**.

Each primitive is a learned controller. Sequences are planned by a grammar parser. New skills are composed like sentences.

**Why revolutionary:** Makes dexterous manipulation **structured, interpretable, and transferable** across hands and tasks.

**Architecture:**
* Library of contact primitives (each = small policy + precondition/postcondition).
* Symbolic grammar for valid primitive sequences.
* Parser: given task, generate contact "sentence."
* Executer: run primitives with reactive feedback.

**Blockers solved:** Dexterous manipulation, long-horizon tasks, interpretability.

**Feasibility:** Medium. Start with a 3-finger gripper in MuJoCo and 5 primitives.

---

### Concept I: Uncertainty-Aware Autonomous Robot

**The problem:** Robots are either overconfident or excessively cautious. They don't know what they don't know, and they can't ask for help.

**The idea:** A robot that maintains calibrated uncertainty over:
* World state (where things are).
* Dynamics parameters (mass, friction).
* Policy competence (will my policy succeed here?).
* Task understanding (did I interpret the instruction correctly?).

It asks for help or slows down when uncertainty is high.

**Why revolutionary:** Trustworthy autonomy. The robot becomes a rational agent that knows its limits.

**Architecture:**
* Bayesian state estimation + ensemble dynamics models.
* Conformal prediction for success probability.
* Active learning module queries human when information gain is high.
* Risk-sensitive planner (CVaR / robust MPC).

**Blockers solved:** Safety, personalization, deployment in unstructured homes.

**Feasibility:** High. Many components are mature. Can be built on top of any base policy.

---

### Concept J: The 100-Robot Skill-Sharing Network

**The problem:** Every robot learns in isolation. Useful skills are rediscovered thousands of times.

**The idea:** A federated network where cheap robots in different homes/factories learn locally, extract **skill embeddings**, and share them in a common latent space. A robot in Sydney benefits from a robot in Berlin learning to open a particular jar.

**Why revolutionary:** Democratizes robot skill acquisition. Creates a **collective robot intelligence** without centralizing raw sensor data (privacy-preserving).

**Architecture:**
* Local robot trains skill-specific policies.
* Skill encoder compresses policy + context into embedding.
* Federated server clusters skills and distributes relevant embeddings.
* Local robot retrieves nearest skill and adapts quickly.

**Blockers solved:** Data scarcity, cost, personalization, generalization.

**Feasibility:** Medium. Requires community of robots or simulated multi-user environment.

---

### Concept K: Wearable Teleoperation for Robot Teachers

**The problem:** Existing teleoperation (VR controllers, GELLO, ALOHA) is either expensive, unnatural, or slow. Good robot teachers are rare.

**The idea:** A cheap wearable system (gloves + phone camera + IMU) that lets anyone naturally demonstrate manipulation tasks. The data is automatically converted into robot-executable demonstrations.

**Why revolutionary:** Makes high-quality robot demonstration data as cheap as recording a video.

**Architecture:**
* Finger-tracking glove (~$50 using U-ARM approach).
* Phone/ webcam captures third-person view.
* Hand-object pose estimation.
* Retargeting to robot hand/arm in real time.
* Auto-labeling of contact phases and task segments.

**Blockers solved:** Data collection cost, accessibility.

**Feasibility:** Very High. Hardware can be built for under $100. Software stack can leverage existing pose estimators.

---

### Concept L: Robotic "Physical Intuition" Benchmark (PIBench) — Phases 0–2 Complete

**The problem:** There is no standardized, model-agnostic, causal way to measure whether a robot (or model) understands physical common sense. Existing benchmarks test passive video QA, 2D puzzles, or task-specific robot planning — not the underlying physical concepts.

**The idea:** Create an open benchmark of physical reasoning tasks that can evaluate analytical functions, neural nets, VLMs, or humans. Each scene exposes latent physical parameters (mass, friction, CoM, shape) so a failure tells you *which concept* the model does not understand.

**PIBench is now being built.** Current codename: `pibench/` in the `LearningRobotics` repo.

**Phase 0 — Engine scaffold (complete):**
* Model-agnostic harness: `Problem`, `Suite`, `Runner`, `Evaluator`, `Predictor`.
* First scene: `TowerFall` (statics — which tower falls when the platform tilts?).
* Baselines: `physics_oracle` (100% on deterministic scenes) and `random`.
* CLI: `pibench list`, `pibench run`, `pibench render`.
* Tests: 7 passing.

**Phase 1 — Statics suite (complete):**
* `SlopeSlide` — does a block slide down an incline? (`tan(θ) > μ_s`).
* `SupportBalance` — balance point of an asymmetric loaded beam.
* `ToppleDirection` — which way does an off-center stack topple?

**Phase 2 — Dynamics suite (complete):**
* `PendulumSwing` — estimate small-angle period `T ≈ 2π√(L/g)`.
* `CollisionBounce` — 1D elastic collision outcome.
* `ProjectileHit` — predict range `R = v² sin(2θ)/g`.

**Planned phases:**

| Phase | Suite | What it tests |
|---|---|---|
| **3** | Contact & friction | Tip vs slide, stack stability, slip, grip |
| **4** | Articulated & deformable | Joints, constraints, ropes, cloth, gears |
| **5** | Parameter estimation & counterfactuals | Mass/friction ordering, "what if mass doubled?" |
| **6** | Model evaluation harness + leaderboard | VLM stub, calibration metrics, static HTML leaderboard |
| **7** | Real-robot validation subset | Validation protocol + harness proven on a realistic virtual arm; AM-ARM / Forte adapters are optional future work |

**Why revolutionary:** Standardization accelerates the field, but *causal* standardization redirects research toward the actual concepts robots are missing. PIBench also proves that a single developer on a laptop can ship a credible benchmark.

**Blockers solved:** Evaluation fragmentation; model-agnostic physical-reasoning evaluation; diagnostic failure analysis.

**Feasibility:** Very High. Built entirely in MuJoCo; first scene already runs on the RTX 5060 laptop.

**Run it now:**

```powershell
cd C:\Users\point\projects\LearningRobotics\pibench
. .venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Users\point\projects\LearningRobotics\pibench"
python -m pibench run --suite statics --predictor physics_oracle --n 10
python -m pibench run --suite dynamics --predictor physics_oracle --n 10
python run_all.py
```

---

## 5. Impact-Domain Map

| Real-world domain | Top unsolved need | Best-matched concept |
|---|---|---|
| **Elderly care at home** | Reliable, personalized assistance in unstructured homes | I + F + K |
| **Construction labor shortage** | Autonomous, robust, contact-aware work in dynamic sites | A + D + H |
| **Warehouse logistics** | Generalize to unseen objects; handle deformable/fragile items | C + E + J |
| **Industrial assembly** | Precision insertion, long-horizon tasks, fast reprogramming | B + F + H |
| **Education / hobby robotics** | Cheap hardware + accessible programming | K + L |
| **Developing-world automation** | Low-cost, robust robots that work without perfect infrastructure | K + J + I |

---

## 6. Recommended Attack Plan (for this repo)

Given our current setup — **MuJoCo, RTX 5060, Chapter 2 next** — here is a pragmatic progression that builds toward something revolutionary:

### Phase 0: Foundations (now — Chapters 2–5)
* Master rigid-body transforms, forward/inverse kinematics, Jacobians, dynamics.
* Build reusable MuJoCo robot models.
* **PIBench (Concept L) Phases 0–2 are complete.** Engine scaffold + statics + dynamics suites done; continue with contact/articulated scenes while reading Chapter 4.

### Phase 1: Skill Representation (Chapters 6–7)
* Build a library of contact primitives in MuJoCo (Concept H seed).
* Implement constraint-language programming for simple tasks (Concept F seed).
* Add uncertainty-aware control wrappers (Concept I seed).

### Phase 2: Learning from Observation (Chapter 8+)
* Video → sim reconstruction for simple tabletop tasks (Concept A prototype).
* Train affordance maps from internet video or synthetic data (Concept E prototype).
* Build the wearable teleop glove if budget allows (Concept K).

### Phase 3: Virtual Real-Sim-Real Loop
* Build a realistic virtual arm with configurable actuator/sensor dynamics and domain randomization.
* Implement self-improving calibration loop (Concept D) entirely in simulation.
* Publish reproducible simulation results and open-source everything.
* Optional future step: swap in a real AM-ARM / Forte adapter and rerun the same loop.

### Phase 4: Foundation-Model Integration
* Add LLM/VLM planning with physics verification (Concept B).
* Build causal skill manifolds (Concept C).
* Connect to the 100-robot network idea (Concept J).

---

## 7. The One-Line North Star

> **Build a robot that learns physical skills from ordinary video, reasons about them with a physics engine, executes them with provable safety, and shares what it learns with other robots — for effectively $0 in hardware by proving the full loop in simulation first, with an optional sub-$500 physical arm as a future extension.**

That is extraordinary, revolutionary, and technically possible from where we stand.

---

## 8. Sources & Further Reading

### VLA / Foundation Models for Robotics
* [DexSim2Real: Foundation Model-Guided Sim-to-Real Transfer](https://arxiv.org/html/2605.05241)
* [ABot-M0: VLA Foundation Model with Action Manifold Learning](https://arxiv.org/html/2602.11236)
* [Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement](https://www.microsoft.com/en-us/research/articles/object-centric-residual-rl/)
* [Are Foundation Models the Route to Full-Stack Transfer in Robotics?](https://arxiv.org/pdf/2602.22001)
* [Gemini Robotics: Bringing AI into the Physical World](https://doi.org/10.48550/arxiv.2503.20020)
* [Vesta: A Generalist Embodied Reasoning Model](https://doi.org/10.48550/arxiv.2606.20905)
* [Embodied-R1.5: Evolving Physical Intelligence](https://arxiv.org/html/2606.11324)

### Physics-Grounded / Causal World Models
* [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998v2)
* [Object-Centric World Models for Causality-Aware RL](https://ojs.aaai.org/index.php/AAAI/article/view/39642)
* [DexWorldModel: Causal Latent World Modeling](https://arxiv.org/html/2604.16484)
* [World4RL: Diffusion World Models for Policy Refinement](https://arxiv.org/html/2509.19080v2)
* [PhysVLA: Towards Physically-Grounded VLA](https://arxiv.org/html/2606.13886)
* [SIMPACT: Simulation-Enabled Action Planning using VLMs](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_SIMPACT_Simulation-Enabled_Action_Planning_using_Vision-Language_Models_CVPR_2026_paper.html)

### Differentiable Simulation & Sim-to-Real
* [Closing Sim-to-Real Gap via Differentiable Simulation (HALO)](https://arxiv.org/abs/2603.15084v1)
* [Real-Sim-Real (RSR) Loop with Differentiable Simulation](https://arxiv.org/html/2503.10118v2)
* [Hard Contacts with Soft Gradients (DiffMJX)](https://arxiv.org/html/2506.14186v1)
* [D-REX: Differentiable Real-to-Sim-to-Real for Dexterous Grasping](https://arxiv.org/html/2603.01151)
* [Learning Deployable Locomotion Control via Differentiable Simulation](https://proceedings.mlr.press/v305/schwarke25a.html)

### Contact-Rich Manipulation
* [From a Single Demonstration to a General Policy for Contact-Rich Manipulation](https://arxiv.org/html/2605.17601)
* [FA-RDP: Frequency-Adaptive Reactive Diffusion Policy](https://arxiv.org/html/2607.28596)
* [PhaForce: Phase-Scheduled Visual–Force Policy Learning](https://arxiv.org/html/2603.08342v1)
* [Data and Learning Where it Matters for Contact-Rich Manipulation](https://arxiv.org/html/2607.15982)
* [Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning](https://arxiv.org/html/2604.23609v1)

### Neuro-Symbolic Planning
* [H-WM: Robotic TAMP Guided by Hierarchical World Model](https://arxiv.org/abs/2602.11291v2)
* [Learning Sound Symbolic Abstractions from VLMs for TAMP](https://openreview.net/forum?id=liVUIlgUI5)
* [Fast Task Planning With Neuro-Symbolic Relaxation (Flax)](https://doi.org/10.1109/lra.2026.3662556)
* [LLM-Flax: Generalizable Robotic Task Planning](https://arxiv.org/html/2604.26569)
* [Neuro-Symbolic Learning for Long-Horizon Task Planning (iFlax)](https://arxiv.org/html/2606.06877)

### Unsolved Problems / Real-World Domains
* [Open Problems in Robot Learning: 10 Research Directions for 2025–2028](https://www.roboticscenter.ai/research/robot-learning-research-directions-2025)
* [The Reality Gap in Robotics: Challenges, Solutions, and Best Practices](https://www.annualreviews.org/content/journals/10.1146/annurev-control-031924-100130)
* [Where Autonomy Works: Evaluating Robot Capabilities in 2026](https://epoch.ai/publications/where-autonomy-works-evaluating-robot-capabilities-in-2026)
* [Construction Robotics Report 2026 - Zacua Ventures](https://zacuaventures.com/construction-robotics-report-2026/)
* [2026 Jobsite Robotics Tech Specialty Report - BuiltWorlds](https://builtworlds.com/insights/2026-equipment-robotics-jobsite-robotics-tech-specialty-report/)
* [The Deployment and Use of Social Robots for Home-Based Healthcare](https://link.springer.com/article/10.1007/s12369-026-01391-1)

### Low-Cost Open Hardware
* [AM-ARM: low-cost 6+1 DoF arm](https://github.com/liyiteng/AM-ARM)
* [Forte: Strong, Accurate, Low-Cost Manipulator](https://arxiv.org/html/2507.15693)
* [Low-Cost Robot Arm](https://github.com/AlexanderKoch-Koch/low_cost_robot)
* [Open Arms Mini](https://github.com/pkooij/open-arms-mini)
* [U-ARM: Ultra low-cost teleoperation interface](https://arxiv.org/html/2509.02437)
* [Zeroth-01 Bot: open-source humanoid](https://github.com/zeroth-robotics/zeroth-bot)
* [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm/)

---

*Generated 2026-08-13. This document is a living artifact — update it as ideas evolve and experiments run.*
