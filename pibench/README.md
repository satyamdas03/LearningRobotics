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
  statics: ['TowerFall']

Predictor: physics_oracle
Overall accuracy: 100.0% (5/5)

Per-suite accuracy:
  statics             : 100.0% (5/5)
```

---

## 📦 Install from scratch

```bash
cd pibench
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dependencies: `mujoco>=3.11.0`, `numpy`, `pydantic`, `pillow`, `pytest`.

---

## 🏗️ Repository layout

```
pibench/
├── pibench/                    # Python package
│   ├── cli.py                  # `python -m pibench ...`
│   ├── core/                   # Engine: Problem, Suite, Runner, Evaluator, Registry
│   ├── scenes/                 # MuJoCo scene implementations
│   │   ├── statics/
│   │   │   └── tower_fall.py   # Phase 0 scene
│   │   ├── dynamics/           # (Phase 2)
│   │   ├── contact/            # (Phase 3)
│   │   ├── articulated/        # (Phase 4)
│   │   └── params/             # (Phase 5)
│   ├── predictors/             # Baselines and model stubs
│   │   ├── base.py
│   │   ├── random_predictor.py
│   │   └── physics_oracle.py
│   └── utils/
│       └── mjcf.py             # Programmatic MJCF helpers
├── tests/
│   └── test_core.py            # Engine + tower_fall tests
├── run_all.py                  # Run all suites across all baselines
├── requirements.txt
└── README.md                   # This file
```

---

## 🧪 Running baselines

```powershell
# Physics oracle (upper bound)
python -m pibench run --suite statics --predictor physics_oracle --n 20

# Random baseline
python -m pibench run --suite statics --predictor random --n 20 --seed 0

# Save results to JSON
python -m pibench run --suite statics --predictor physics_oracle --n 20 --output output/results_oracle.json

# Render a scene thumbnail
python -m pibench render TowerFall --seed 0 --output output/tower_fall_seed0.png
```

---

## 🧩 The first scene: `TowerFall`

**Suite:** `statics`  
**Concepts:** center of mass, support polygon, stability  
**Question:** *Two towers stand on a tilting platform. Which tower falls?*

Two towers are generated with randomized dimensions:

* **Tower A:** narrow and tall (unstable)
* **Tower B:** wide and short (stable)

The ground truth is computed by a MuJoCo rollout: after the platform tilts, whichever tower's blocks fall below a height threshold is the loser. The predictor must answer `"A"` or `"B"`.

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

Current status: **7 passed**.

---

## 📊 Roadmap

| Phase | Suite | Textbook alignment | Target |
|---|---|---|---|
| **0** | Engine + `tower_fall` | Foundations | ✅ runnable package |
| **1** | Statics & stability | Chapter 2 (rigid-body motions, CoM) | ≥4 scenes |
| **2** | Dynamics | Newtonian mechanics | ≥5 scenes |
| **3** | Contact & friction | Chapter 4 (velocity kinematics / contacts) | ≥5 scenes |
| **4** | Articulated & deformable | Chapter 5 (IK / constraints) | ≥5 scenes |
| **5** | Parameter estimation & counterfactuals | Consolidation + system ID | ≥5 scenes |
| **6** | Model harness + leaderboard | — | VLM stub, static leaderboard |
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
