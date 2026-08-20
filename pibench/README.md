# 🧱 PIBench — Physical Intuition Benchmark

> A lightweight, open-source, **MuJoCo-based** benchmark for evaluating physical common-sense reasoning in AI models, robot policies, and humans.

PIBench is the first concrete deliverable from the *Revolutionary Robotics* manifesto (Concept L). It is built in parallel with the textbook curriculum (Chapters 2–5) and is designed to run entirely on a laptop with an RTX 5060 — no cloud GPU required.

---

## ✨ What makes PIBench different

| Feature | Why it matters |
|---|---|
| **Model-agnostic** | Plug in an analytical function, a neural net, a VLM API, or a human. |
| **Causal / diagnostic** | Every scene exposes latent physical parameters (mass, friction, CoM, shape) so failures can be traced to a concept. |
| **Curriculum-aligned** | Suites map to textbook chapters: statics → dynamics → contact → articulated/deformable → parameter estimation. |
| **Pure MuJoCo + Python** | No rendering farm, no proprietary sim, no massive assets. |
| **Reproducible by default** | Deterministic seeds, pinned dependencies, and a static leaderboard generator. |

---

## 🚀 Quick start

On Windows (PowerShell):

```powershell
cd C:\Users\point\projects\LearningRobotics\pibench
. .venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\Users\point\projects\LearningRobotics\pibench"
python -m pibench list --suites
python -m pibench run --suite statics --predictor physics_oracle --n 5
```

On Linux/macOS:

```bash
cd pibench
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
python -m pibench list --suites
python -m pibench run --suite statics --predictor physics_oracle --n 5
```

Expected output:

```text
Registered suites:
  articulated: ['DoorSwing', 'DrawerPull', 'GearTurn', 'RopeTension']
  contact: ['FrictionPile', 'PushTipVsSlide', 'SlipGrip', 'StackStability', 'WedgeInsert']
  deformable: ['ChainDrape']
  dynamics: ['CollisionBounce', 'PendulumSwing', 'ProjectileHit']
  params: ['BalanceAfterMove', 'CounterfactualFriction', 'CounterfactualMass', 'FrictionOrder', 'MassOrder']
  statics: ['SlopeSlide', 'SupportBalance', 'ToppleDirection', 'TowerFall']

Predictor: physics_oracle
Overall accuracy: 100.0% (5/5)

Per-suite accuracy:
  articulated         : 100.0% (5/5)
```

---

## 📦 Install from scratch

```bash
cd pibench
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dependencies: `mujoco>=3.11.0`, `numpy`, `pydantic`, `pillow`, `pytest`. Optional: `anthropic>=0.30.0` for the LLM predictor.

---

## 🏗️ Repository layout

```
pibench/
├── pibench/                    # Python package
│   ├── cli.py                  # `python -m pibench ...`
│   ├── core/                   # Engine: Problem, Suite, Runner, Evaluator, Registry
│   ├── scenes/                 # MuJoCo scene implementations
│   │   ├── statics/
│   │   │   ├── slope_slide.py  # Phase 1: friction angle
│   │   │   ├── support_balance.py  # Phase 1: torque balance
│   │   │   ├── topple_direction.py # Phase 1: toppling direction
│   │   │   └── tower_fall.py   # Phase 0 scene
│   │   ├── dynamics/
│   │   │   ├── collision_bounce.py # Phase 2: elastic collision
│   │   │   ├── pendulum_swing.py   # Phase 2: pendulum period
│   │   │   └── projectile_hit.py   # Phase 2: projectile range
│   │   ├── contact/            # Phase 3: contact & friction
│   │   │   ├── friction_pile.py      # Hardest object to push
│   │   │   ├── push_tip_vs_slide.py  # Tip or slide?
│   │   │   ├── slip_grip.py          # Lift or slip?
│   │   │   ├── stack_stability.py    # Survive a side tap?
│   │   │   └── wedge_insert.py       # Fit or jam?
│   │   ├── articulated/        # Phase 4: articulated constraints
│   │   │   ├── drawer_pull.py        # Pull drawer: does it open?
│   │   │   ├── door_swing.py         # Push door: does it swing?
│   │   │   ├── gear_turn.py          # Which way does the meshed gear turn?
│   │   │   └── rope_tension.py       # Which side of the pulley descends?
│   │   ├── deformable/         # Phase 4: coarse deformable bodies
│   │   │   └── chain_drape.py        # Free-end height of a chain over a bar
│   │   └── params/                # Phase 5: parameter estimation & counterfactuals
│       ├── mass_order.py         # Order blocks by mass from observed displacements
│       ├── friction_order.py     # Rank surfaces by slipperiness from tilt threshold
│       ├── counterfactual_mass.py     # What if top mass doubled?
│       ├── counterfactual_friction.py   # What if friction were zero?
│       └── balance_after_move.py        # Support shift after moving a point mass
│   ├── predictors/             # Baselines and model stubs
│   │   ├── base.py
│   │   ├── random_predictor.py
│   │   ├── physics_oracle.py
│   │   └── llm_predictor.py   # Optional Anthropic API predictor (cache + random fallback)
│   └── utils/
│       ├── mjcf.py             # Programmatic MJCF helpers
│       ├── contact.py          # Contact-event and pusher helpers
│       └── articulated.py      # Joints, tendons, capsule chains
├── tests/
│   └── test_core.py            # Engine + all scenes (55 tests)
├── run_all.py                  # Run all suites across all baselines
├── requirements.txt
└── README.md                   # This file
```

---

## 🧪 Running baselines

```powershell
# Physics oracle (upper bound) on every suite
python -m pibench run --suite statics --predictor physics_oracle --n 20
python -m pibench run --suite dynamics --predictor physics_oracle --n 20
python -m pibench run --suite contact --predictor physics_oracle --n 20
python -m pibench run --suite articulated --predictor physics_oracle --n 20
python -m pibench run --suite deformable --predictor physics_oracle --n 20
python -m pibench run --suite params --predictor physics_oracle --n 20

# Random baseline
python -m pibench run --suite statics --predictor random --n 20 --seed 0
python -m pibench run --suite dynamics --predictor random --n 20 --seed 0
python -m pibench run --suite contact --predictor random --n 20 --seed 0
python -m pibench run --suite articulated --predictor random --n 20 --seed 0
python -m pibench run --suite deformable --predictor random --n 20 --seed 0
python -m pibench run --suite params --predictor random --n 20 --seed 0

# Optional LLM baseline (requires anthropic SDK + ANTHROPIC_API_KEY)
python -m pibench run --suite params --predictor llm --n 5

# Save results to JSON
python -m pibench run --suite contact --predictor physics_oracle --n 20 --output output/results_oracle.json

# Render a scene thumbnail
python -m pibench render TowerFall --seed 0 --output output/tower_fall_seed0.png
python -m pibench render PushTipVsSlide --seed 0 --output output/push_tip_vs_slide_seed0.png
python -m pibench render ProjectileHit --seed 0 --output output/projectile_hit_seed0.png
python -m pibench render DrawerPull --seed 0 --output output/drawer_pull_seed0.png

# Open any scene in the interactive MuJoCo viewer
python -m pibench view TowerFall            # static inspection
python -m pibench view PendulumSwing --simulate  # watch it run live
python -m pibench view DrawerPull --simulate

# Render a full visual showcase of every chapter + PIBench scene
python showcase.py                 # writes output/showcase/*.png
python build_showcase_artifact.py  # writes output/showcase/index.html

# Run all baselines across all suites
python run_all.py
```

---

## 🧩 Scene examples

### `TowerFall` (statics)

**Suite:** `statics`  
**Concepts:** center of mass, support polygon, stability  
**Question:** *Two towers stand on a tilting platform. Which tower falls?*

Two towers are generated with randomized dimensions:

* **Tower A:** narrow and tall (unstable)
* **Tower B:** wide and short (stable)

The ground truth is computed by a MuJoCo rollout: after the platform tilts, whichever tower's blocks fall below a height threshold is the loser. The predictor must answer `"A"` or `"B"`.

### `PendulumSwing` (dynamics)

**Concepts:** pendulum period, length scaling, mass independence  
**Question:** *Two pendulums are released from the same angle. Which has the longer period?*

The analytic ground truth is `T ≈ 2π√(L/g)`, so the longer pendulum wins regardless of mass.

### `ProjectileHit` (dynamics)

**Concepts:** projectile motion, range equation  
**Question:** *A ball is launched from ground level at a given speed and angle. How far does it land?*

The analytic range `R = v² sin(2θ) / g` is the answer, validated against a MuJoCo rollout.

### `SlipGrip` (contact)

**Concepts:** Coulomb friction, grip force, weight  
**Question:** *A parallel-jaw gripper squeezes a block and lifts it. Does the block lift or slip?*

The analytic ground truth compares the total available friction `2 * mu * F_grip` to the block's weight `m * g`.

### `DrawerPull` (articulated)

**Concepts:** prismatic joints, static friction, actuators  
**Question:** *A drawer is pulled with a constant force. Does it open or jam?*

The drawer sits on a slide joint with `frictionloss`. A motor actuator applies the pull force, and the ground truth is yes/no based on whether the displacement exceeds a threshold.

### `RopeTension` (articulated)

**Concepts:** tension, pulleys, constrained motion, gravity  
**Question:** *Two masses hang from a rope over a pulley. Which side descends?*

Two free bodies are linked by a MuJoCo spatial tendon passing over a pulley site. The heavier side descends; if masses are close, the outcome is `same`.

### `ChainDrape` (deformable)

**Concepts:** deformable-body approximation, contact, sag geometry  
**Question:** *A chain is draped over a horizontal bar. How high is the free end above the floor?*

A coarse deformable approximation is built from nested capsules connected by ball joints. The numeric answer is the final height of the chain's free end after settling.

### `MassOrder` (params)

**Concepts:** Mass, impulse, displacement inference  
**Question:** *Three blocks receive identical pushes on a frictionless surface. Which is heaviest?*

The heaviest block experiences the smallest displacement, so the answer follows the inverse of observed displacement.

### `FrictionOrder` (params)

**Concepts:** Static friction, tilt threshold, surface ranking  
**Question:** *Three blocks sit on a tilting platform with different surface friction. Which surface is most slippery?*

The block with the lowest static-friction coefficient slides first, so its surface is the slipperiest.

### `CounterfactualMass` (params)

**Concepts:** Counterfactual reasoning, stability, mass scaling  
**Question:** *A tower stands with a top block. If the top block's mass were doubled, would the tower still stand?*

The scene rebuilds itself with the doubled mass and reruns the tilt simulation to answer yes/no.

### `BalanceAfterMove` (params)

**Concepts:** Torque balance, support shift, center of mass  
**Question:** *A point mass on a beam is moved a known distance. How far must the support shift to keep the beam balanced?*

The answer comes from an analytic support-location update; the scene verifies it numerically.

---

## 🧰 Adding a new scene

1. Create a Python file under `pibench/scenes/<suite>/`.
2. Inherit from `pibench.core.problem.Problem`.
3. Implement `_build_scene()`, `question()`, `ground_truth()`, and `score()`.
4. Decorate the class with `@register_problem("suite_name")`.
5. Add a test in `tests/`.

Example skeleton:

```python
from pibench.core.problem import Problem, Question, AnswerType, GroundTruth, Prediction
from pibench.core.registry import register_problem

@register_problem("statics")
class MyScene(Problem):
    def _build_scene(self) -> None:
        # Build MJCF / MuJoCo state from self.seed
        ...

    def question(self) -> Question:
        return Question(
            text="Which object falls first?",
            answer_type=AnswerType.CHOICE,
            choices=["A", "B"],
        )

    def ground_truth(self) -> GroundTruth:
        # Run MuJoCo or analytic computation
        return GroundTruth(answer="A", explanation="Center of mass leaves support polygon.")

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
```

---

## 🧪 Running tests

```powershell
$env:PYTHONPATH = "C:\Users\point\projects\LearningRobotics\pibench"
python -m pytest tests -q
```

Current status: **55 passed**.

---

## 📊 Roadmap

| Phase | Suite | Textbook alignment | Target |
|---|---|---|---|
| **0** | Engine + `tower_fall` | Foundations | ✅ runnable package |
| **1** | Statics & stability | Chapter 2 (rigid-body motions, CoM) | ✅ 4 scenes |
| **2** | Dynamics | Newtonian mechanics | ✅ 3 scenes |
| **3** | Contact & friction | Chapter 4 (velocity kinematics / contacts) | ✅ 5 scenes |
| **4** | Articulated & deformable | Chapter 5 (IK / constraints) | ✅ 5 scenes |
| **5** | Parameter estimation & counterfactuals | Consolidation + system ID | ✅ 5 scenes |
| **6** | Model harness + leaderboard | — | VLM/LLM harness, static leaderboard |
| **7** | Real-robot validation subset | — | Protocol for AM-ARM / Forte |

---

## 🤝 Contributing to PIBench

1. Pick a physical concept from the current textbook chapter or the manifesto blockers.
2. Propose a scene in a GitHub issue or in `docs/SCENE_CATALOG.md`.
3. Implement with a deterministic seed, a clear question, and a MuJoCo/analytic ground truth.
4. Add a test and a short scene card.
5. Open a PR.

See `docs/CONTRIBUTING.md` (Phase 1) for the full style guide.

---

## 🔗 Links

* Main repo: [satyamdas03/LearningRobotics](https://github.com/satyamdas03/LearningRobotics)
* Revolutionary manifesto: [`REVOLUTIONARY_ROBOTICS_IDEAS.md`](../REVOLUTIONARY_ROBOTICS_IDEAS.md)
* Project memory: [`memory.md`](../memory.md)

---

**License:** MIT — use it, fork it, improve it.

> *"If a model can't predict which tower falls, it doesn't understand physics — it only remembers textures."* — PIBench
