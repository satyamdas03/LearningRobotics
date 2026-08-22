# Plan: Physical Intuition Benchmark (PIBench) — Phase 0

## Goal

Build **PIBench** (Physical Intuition Benchmark): a lightweight, open-source, MuJoCo-based benchmark for evaluating whether an AI model, robot policy, or human understands physical common sense. It is the first concrete deliverable from the "Revolutionary Robotics" manifesto (Concept L) and will be constructed in parallel with textbook Chapters 2–5 (rigid-body motions through inverse kinematics).

The benchmark must be:

* **Runnable on the RTX 5060 laptop** — no cloud GPU required for development.
* **Model-agnostic** — evaluate analytical functions, neural nets, VLMs via API, or humans.
* **Causal and diagnostic** — each scene exposes latent physical parameters (mass, friction, CoM, shape) so failures can be traced to concepts.
* **Aligned with the curriculum** — every suite exercises the theory from the current textbook chapter.
* **Public and reproducible** — committed to `satyamdas03/LearningRobotics` with a leaderboard-ready static dashboard.

---

## Differentiation: Why PIBench vs. Existing Benchmarks

Existing work is excellent but has gaps PIBench will fill:

| Benchmark | What it tests | Gap PIBench fills |
|---|---|---|
| **IntPhys / IntPhys 2** | Intuitive physics from video (object permanence, solidity) | No active embodied control; no parameter/counterfactual queries. |
| **PHYRE / CRAFT** | 2D puzzle-like physical reasoning | Not robot-control oriented; limited contact diversity. |
| **CLEVRER / CATER** | Object dynamics, occlusion, collisions | Passive video QA; no agent action or sim-to-real bridge. |
| **KinDER (2026)** | MuJoCo robot learning/planning with 25 envs | Focused on planning baselines, not model-agnostic physical-intuition evaluation with causal factors. |
| **MuBlE / SHOP-VRB2** | MuJoCo + Blender manipulation planning | Heavy Blender rendering dependency; not lightweight laptop-first. |
| **WorldBench / PAI-Bench** | World-model / video-generation diagnostics | Not a runnable embodied-control benchmark with a simple harness. |
| **RoboDojo / VLA-Replica** | VLA sim-to-real evaluation | Requires Isaac Sim or real hardware; too heavy for our phase. |

**PIBench's unique angle:**

1. **Pure MuJoCo, pure Python** — no rendering farm, no massive assets, no proprietary sim.
2. **Latent-factor control** — every scene is procedurally generated with randomized but recorded physical parameters, enabling causal analysis.
3. **Counterfactual queries** — ask "what if mass were doubled?" not only "what happens next?".
4. **Curriculum-aligned suites** — statics → dynamics → contact → articulated/deformable → parameter estimation, matching the textbook.
5. **Built-in model-agnostic harness** — plug in a VLM, a diffusion world model, a physics oracle, or a human.
6. **Static dashboard + leaderboard** — auto-generated GitHub Pages site with scene cards, metrics, and a leaderboard.

---

## Repository Layout (to create)

```
LearningRobotics/
├── README.md
├── memory.md
├── REVOLUTIONARY_ROBOTICS_IDEAS.md
├── .gitignore
├── chapter01_foundation/
├── chapter02_rigid_body_motions/
├── chapter03_forward_kinematics/
├── chapter04_velocity_kinematics/
├── chapter05_inverse_kinematics/
└── pibench/                                 # NEW — Physical Intuition Benchmark
    ├── README.md                            # PIBench overview, quickstart, leaderboard link
    ├── requirements.txt                     # mujoco, numpy, pydantic, pytest, jinja2, pillow
    ├── .venv/                               # (ignored) local Python 3.11 venv
    ├── pibench/                             # Python package
    │   ├── __init__.py
    │   ├── cli.py                           # `pibench run`, `pibench list`, `pibench render`, `pibench leaderboard`
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── scene.py                     # Abstract base for a PIBench scene
    │   │   ├── problem.py                   # Question, answer, scoring logic
    │   │   ├── suite.py                     # Collection of problems
    │   │   ├── runner.py                    # Execute problems against a predictor
    │   │   ├── evaluator.py                 # Compute metrics per suite / overall
    │   │   ├── renderer.py                  # MuJoCo → RGB image (optional, for dashboard)
    │   │   └── registry.py                  # Auto-discover all scene modules
    │   ├── scenes/                          # MJCF assets + scene implementations
    │   │   ├── __init__.py
    │   │   ├── _assets/                     # Shared MuJoCo assets (plane, lights, primitives)
    │   │   ├── statics/
    │   │   │   ├── __init__.py
    │   │   │   ├── tower_fall.py            # Which tower falls? stability / CoM
    │   │   │   ├── slope_slide.py           # Does object slide or stay? friction angle
    │   │   │   └── support_balance.py       # Where to place support so beam balances?
    │   │   ├── dynamics/
    │   │   │   ├── __init__.py
    │   │   │   ├── collision_bounce.py      # After collision, which object moves faster?
    │   │   │   ├── pendulum_swing.py        # Which pendulum swings higher? mass/length
    │   │   │   └── projectile_hit.py        # Where does the projectile land?
    │   │   ├── contact/
    │   │   │   ├── __init__.py
    │   │   │   ├── push_tip_vs_slide.py     # Pushed block: tip or slide?
    │   │   │   ├── stack_stability.py       # Will stack survive a tap?
    │   │   │   ├── wedge_insert.py          # Can wedge fit into gap?
    │   │   │   ├── friction_pile.py       # Which object is hardest to push?
    │   │   │   └── slip_grip.py             # Gripper lifts or slips?
    │   │   ├── articulated/
    │   │   │   ├── __init__.py
    │   │   │   ├── drawer_pull.py           # Pull drawer: does it open fully?
    │   │   │   ├── door_swing.py            # Which way does door swing under push?
    │   │   │   ├── rope_tension.py          # Which rope segment breaks first?
    │   │   │   └── gear_turn.py             # Which way does the meshed gear turn?
    │   │   ├── deformable/
    │   │   │   ├── __init__.py
    │   │   │   └── chain_drape.py           # Free-end height of chain over bar
    │   │   └── params/
    │   │       ├── __init__.py
    │   │       ├── mass_order.py            # Which object is heaviest?
    │   │       ├── friction_order.py        # Which surface is most slippery?
    │   │       └── counterfactual_mass.py   # What if mass doubled?
    │   ├── predictors/
    │   │   ├── __init__.py
    │   │   ├── base.py                      # Predictor interface
    │   │   ├── random.py                    # Random-guess baseline
    │   │   ├── physics_oracle.py            # MuJoCo rollout baseline (upper bound)
    │   │   ├── human.py                     # Interactive CLI predictor
    │   │   └── vlm.py                       # OpenAI/Anthropic VLM predictor stub
    │   ├── metrics/
    │   │   ├── __init__.py
    │   │   └── accuracy.py                  # Per-suite and overall accuracy / calibration
    │   ├── dashboard/
    │   │   ├── __init__.py
    │   │   ├── generate.py                  # Build static leaderboard HTML
    │   │   └── template.html                # Jinja2 template for leaderboard
    │   └── utils/
    │       ├── __init__.py
    │       ├── mjcf.py                      # Helper to build/compose MJCF strings
    │       ├── contact.py                   # Contact-event and pusher helpers
    │       └── articulated.py               # Joint/tendon/capsule-chain helpers
    ├── tests/
    │   ├── __init__.py
    │   ├── test_runner.py
    │   ├── test_scenes.py
    │   └── test_physics_oracle.py
    ├── docs/
    │   ├── SCENE_CATALOG.md                 # One-pager per scene
    │   └── CONTRIBUTING.md                  # How to add a new scene
    ├── output/                              # (ignored) generated images, results JSON, leaderboard
    └── run_all.py                           # Convenience script: run every suite + build leaderboard
```

---

## Phased Implementation

### Phase 0 — Benchmark Engine Scaffold (this plan)

**Goal:** A runnable package with one end-to-end example problem and the harness working.

**Deliverables:**
1. `pibench/` directory and package skeleton.
2. `pibench/requirements.txt` with pinned deps.
3. Virtual environment `pibench/.venv` installed.
4. Abstract `Scene`, `Problem`, `Suite`, `Runner`, `Evaluator`, `Predictor` classes.
5. `RandomPredictor` and `PhysicsOraclePredictor` baselines.
6. One concrete scene: **`tower_fall`** (statics suite).
   * MuJoCo scene with two block towers of different heights/CoM.
   * Question: "Which tower falls when the floor tilts?"
   * Ground truth computed by MuJoCo rollout.
7. CLI: `pibench list`, `pibench run --suite statics --predictor random|physics_oracle`, `pibench render tower_fall`.
8. First tests: runner executes without crashing; physics oracle scores 100% on `tower_fall`.
9. `pibench/README.md` with overview and quickstart.
10. Update root `README.md`, `memory.md`, and `REVOLUTIONARY_ROBOTICS_IDEAS.md` to reflect the new project.
11. Push to GitHub.

**Success criteria:**
* `cd pibench && .venv\Scripts\Activate.ps1 && python -m pytest` passes.
* `python -m pibench run --suite statics --predictor physics_oracle` reports 100% on `tower_fall`.
* `python -m pibench list` shows at least the `statics` suite with `tower_fall`.
* Repo push succeeds.

**Estimated effort:** 1 focused session.

---

### Phase 1 — Statics & Stability Suite

**Textbook alignment:** Chapter 2 (Rigid-Body Motions) + center of mass / support polygon concepts.

**New scenes:**
1. `slope_slide` — predict if a block slides down an incline given angle and friction.
2. `support_balance` — predict where to place a fulcrum so a beam balances.
3. `arch_stability` — predict which arch collapses under load.
4. `topple_direction` — predict which way an off-center tower topples.

**Engine additions:**
* Parameter randomization with deterministic seeds.
* Question types: binary, multiple-choice, numeric.
* Latent-parameter logging to results JSON.

**Success criteria:**
* Statics suite has ≥4 scenes.
* Physics oracle scores ≥95% across suite.
* Random predictor scores near chance.
* Results JSON schema is stable.

---

### Phase 2 — Dynamics Suite

**Textbook alignment:** Chapter 3 (Forward Kinematics is mostly static, but dynamics introduces velocity/acceleration) and basic Newtonian mechanics.

**New scenes:**
1. `collision_bounce` — after collision, which object moves faster? (conservation of momentum)
2. `pendulum_swing` — which pendulum has longer period / higher swing? (length/mass)
3. `projectile_hit` — where does launched ball land? (gravity, velocity)
4. `rolling_race` — which shape rolls faster down a ramp? (sphere vs cylinder vs block)
5. `spin_topple` — spinning top stability vs angular velocity.

**Engine additions:**
* Time-series observation extraction from MuJoCo.
* Numeric-answer scoring with tolerance.

**Success criteria:**
* Dynamics suite has ≥5 scenes.
* Numeric answers use tolerance-based scoring.
* All scenes have MuJoCo rollout ground truth.

---

### Phase 3 — Contact & Friction Suite ✅

**Textbook alignment:** Chapter 4 (Velocity Kinematics / Jacobians) and introduction to contact forces.

**New scenes:**
1. `push_tip_vs_slide` — pushed block tips or slides depending on push height.
2. `stack_stability` — will a stack survive a side tap?
3. `wedge_insert` — can a wedge be driven into a gap?
4. `friction_pile` — which object is hardest to push? (mass + friction)
5. `slip_grip` — gripper lift: object slips or lifts?

Files: `pibench/scenes/contact/{push_tip_vs_slide,stack_stability,wedge_insert,friction_pile,slip_grip}.py`.

**Engine additions:**
* `pibench/utils/contact.py` — prismatic pusher MJCF, mesh-wedge MJCF, contact queries, constant-speed pusher runner, body tilt measurement.
* Contact-event detection (`body_in_contact`, `body_contact_force_norm`).
* Disturbance/action parameterization via velocity-controlled pushers.

**Success criteria:**
* Contact suite has 5 scenes.
* Physics oracle scores 100% on the suite.
* All contact scenes tested in `tests/test_core.py`.

**Status:** completed 2026-08-18.

---

### Phase 4 — Articulated & Deformable Suite ✅

**Textbook alignment:** Chapter 5 (Inverse Kinematics) and introduction to constraints/joints.

**New scenes:**
1. `drawer_pull` — prismatic drawer with `frictionloss`; motor pulls; yes/no opening outcome.
2. `door_swing` — hinge door with `frictionloss`; motor torque; yes/no swing outcome.
3. `rope_tension` — two masses linked by a spatial tendon over a pulley; which descends?
4. `gear_turn` — externally meshed gears; motor drives gear A; which way does B turn?
5. `chain_drape` — coarse deformable capsule chain over a bar; free-end height.

Files:
* Articulated: `pibench/scenes/articulated/{drawer_pull,door_swing,rope_tension,gear_turn}.py`.
* Deformable: `pibench/scenes/deformable/chain_drape.py`.

**Engine additions:**
* `pibench/utils/articulated.py` — MJCF builders for prismatic joints (`mjcf_prismatic`), hinge joints (`mjcf_hinge`), spatial tendons (`mjcf_tendon`), nested capsule chains (`mjcf_capsule_chain`), and runtime helpers (`body_id`, `joint_position`, `body_displacement`).
* Support for equality constraints and spatial tendons.
* Coarse deformable-body approximation via nested capsules connected by ball joints.

**Success criteria:**
* Articulated/deformable suite has ≥5 scenes across both sub-suites.
* Physics oracle scores 100% on deterministic scenes.
* At least one deformable scene runs reliably.

**Status:** completed 2026-08-18.

---

### Phase 5 — Parameter Estimation & Counterfactual Suite ✅

**Textbook alignment:** Consolidates Chapters 2–5; introduces system identification and causal reasoning.

**New scenes:**
1. `mass_order.py` — order objects by mass from observed displacements after identical pushes.
2. `friction_order.py` — rank surfaces by friction from tilt threshold.
3. `counterfactual_mass.py` — "If this object had double mass, would the tower still stand?"
4. `counterfactual_friction.py` — "If friction were zero, would the block slide?"
5. `balance_after_move.py` — "How far must the support shift after a point mass is moved on a beam?"

Files: `pibench/scenes/params/{mass_order,friction_order,counterfactual_mass,counterfactual_friction,balance_after_move}.py`.

**Engine additions:**
* `pibench/core/counterfactual.py` — `CounterfactualBuilder` and `counterfactual(problem, **overrides)` convenience function.
* `pibench/core/problem.py` — `_counterfactual_params()` defaulting to `latent_params` keys; scenes override to control rebuildable parameters.
* Cloning by re-instantiation with the same seed and applying overrides before `_build_scene()`, avoiding unsafe deep-copy of MuJoCo objects.

**Predictor additions:**
* `pibench/predictors/llm_predictor.py` — optional Anthropic API predictor with local cache and random fallback.
* `pibench/cli.py` — dynamic `--predictor` choices; `llm` shown only when `anthropic` is installed.

**Success criteria:**
* Parameter/counterfactual suite has ≥5 scenes. ✅
* Counterfactual API works for mass and friction. ✅
* A model-agnostic predictor harness accepts an optional LLM. ✅

**Status:** completed 2026-08-20.

---

### Phase 6 — Model Evaluation Harness + Leaderboard ✅

**Goal:** Evaluate external models and present results on a static dashboard.

**Deliverables:**
1. `pibench/harness.py` — `EvaluationHarness` wraps `Runner` + result/metrics serialization.
2. `pibench/evaluation/metrics.py` — per-concept accuracy, ECE, Brier score, optional NLL.
3. `pibench/evaluation/leaderboard.py` — `Leaderboard` Pydantic model + static HTML generator.
4. `pibench/predictors/vlm_predictor.py` — `VLMPredictor` using Anthropic vision API with local cache and text-only fallback.
5. `pibench/cli.py` — `pibench leaderboard` command and `--predictor vlm` support.
6. `pibench/pibench/tests/test_evaluation.py` — unit tests for metrics and leaderboard plumbing.

**Still to do (future / Phase 7):**
* `DiffusionWorldModelPredictor` stub for future integration.
* `HumanPredictor` interactive mode.
* GitHub Actions workflow to regenerate leaderboard on push.
* Deploy leaderboard to `gh-pages`.

**Success criteria:**
* `pibench leaderboard --output-dir output` produces `leaderboard.json` + `leaderboard.html`. ✅
* Baseline run (`python run_all.py`) generates both result files and the leaderboard. ✅
* 11 new evaluation tests pass. ✅
* Dashboard pushed to `gh-pages` or served as artifact. ⏳ Future work.

**Status:** completed 2026-08-20.

---

### Phase 7 — Real-Robot Validation Subset ✅

**Goal:** Define a validation protocol and harness that can run on a mocked arm today and transfer to a cheap real arm (AM-ARM / Forte) tomorrow.

**Deliverables:**
1. `pibench/pibench/realrobot/protocol.py` — `ValidationTask`/`ValidationResult` Pydantic models.
2. `pibench/pibench/realrobot/harness.py` — `RealRobotValidationHarness` with mock-arm creation and `reach_q` execution.
3. `pibench/pibench/realrobot/calibration.py` — `ResidualTracker` for online sim-to-real mismatch calibration.
4. `pibench/pibench/tests/test_realrobot.py` — 5 validation harness tests.
5. `pibench/pibench/cli.py` — `pibench validate` command with ASCII `[OK]`/`[FAIL]` markers.

**Success criteria:**
* Validation protocol exists and runs on a mock arm. ✅
* `reach_q` tasks compare predicted vs actual outcomes. ✅
* Residual tracker detects sim-to-real torque mismatch and noise. ✅
* No hardware purchase required in this phase. ✅

**Status:** completed 2026-08-20.

---

## Engineering Decisions

1. **Language:** Python 3.11 (same as Chapter 1 venv base).
2. **Physics engine:** MuJoCo 3.11.0.
3. **Scene composition:** Code-generated MJCF strings + small shared asset library. Avoid huge mesh assets.
4. **Config/data format:** Pydantic models; results as JSONL; leaderboard as JSON + static HTML.
5. **Testing:** pytest for every scene and every predictor.
6. **Rendering:** Optional MuJoCo renderer (`mujoco.Renderer`) for dashboard thumbnails; can be disabled for headless runs.
7. **Predictor interface:** Each predictor receives a `Problem` and returns a `Prediction` with answer + optional reasoning. This makes the benchmark model-agnostic.
8. **Versioning:** Each scene registers itself via a decorator; adding a scene is one Python file + optionally an asset.
9. **Dashboard:** Static HTML generated from JSON results; no backend server needed initially. Later can become a Next.js site if needed.
10. **Git workflow:** Every phase ends with a commit and push to `origin master`.

---

## Frontend / Dashboard Design

The dashboard is intentionally lightweight and static, so it runs on GitHub Pages.

### Pages

1. **Landing page (`index.html`)**
   * Mission statement.
   * Animated hero: rotating tower falling (MuJoCo-rendered GIF).
   * Quick-start code block.
   * Link to leaderboard, scene catalog, GitHub repo.

2. **Scene catalog (`scenes.html`)**
   * Grid of scene cards.
   * Each card: thumbnail GIF, scene name, suite, difficulty, physical concepts tested, "Run this scene" CLI snippet.
   * Filter by suite and concept.

3. **Leaderboard (`leaderboard.html`)**
   * Table: model/predictor name, overall accuracy, per-suite accuracy.
   * Bar charts per suite (inline CSS/SVG, no external JS).
   * Date of evaluation and hardware note.
   * Submit-a-model instructions.

4. **Concept map (`concepts.html`)**
   * Matrix: scenes × physical concepts (stability, friction, mass, momentum, etc.).
   * Shows coverage gaps.

### Visual style

* Clean, technical, light/dark theme-aware.
* Color-coded suites: Statics = blue, Dynamics = green, Contact = orange, Articulated = purple, Deformable = pink, Params = red.
* Monospace fonts for code blocks.
* Responsive CSS grid; no external dependencies (all inline, self-contained).
* Favicon: ⚙️🧠.

### Generation pipeline

```
run_all.py
    ↓
Execute all suites across all predictors
    ↓
Write results.json
    ↓
pibench dashboard generate --results results.json --output docs/_site/
    ↓
Render thumbnails via MuJoCo (optional)
    ↓
GitHub Pages deploy (manual or Actions)
```

---

## Concrete Phase 0 Tasks

| # | Task | File(s) | Owner |
|---|---|---|---|
| 1 | Create `pibench/` package skeleton | `pibench/pibench/*` | us |
| 2 | Add `pibench/requirements.txt` | `pibench/requirements.txt` | us |
| 3 | Install venv and deps | `pibench/.venv/` | us |
| 4 | Implement core abstract classes | `core/scene.py`, `core/problem.py`, `core/suite.py`, etc. | us |
| 5 | Implement predictor interface + baselines | `predictors/base.py`, `predictors/random.py`, `predictors/physics_oracle.py` | us |
| 6 | Implement `tower_fall` scene | `scenes/statics/tower_fall.py` + MJCF asset | us |
| 7 | Implement CLI | `pibench/cli.py` | us |
| 8 | Add tests | `tests/test_*.py` | us |
| 9 | Add PIBench README | `pibench/README.md` | us |
| 10 | Update root docs | `README.md`, `memory.md`, `REVOLUTIONARY_ROBOTICS_IDEAS.md` | us |
| 11 | Push to GitHub | git commit + push | us |

---

## Success Criteria for the Whole Plan

After all phases are complete:

* **≥25 unique scenes** across 6 suites, each with a MuJoCo ground-truth answer.
* **≥3 predictor baselines** (random, physics oracle, human, VLM stub).
* **100% physics-oracle accuracy** on deterministic scenes (validates engine).
* **Static leaderboard/dashboard** auto-generated and shareable.
* **All code tested** with pytest and committed to GitHub.
* **README documents** how to add a scene, run evaluation, and interpret metrics.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| MuJoCo scenes are nondeterministic across OS/versions | Pin MuJoCo version; use fixed seeds; ground truth computed at evaluation time, not precomputed. |
| Rendering is slow on RTX 5060 | Make rendering optional; dashboard can use text + pre-rendered thumbnails only. |
| VLMs via API cost money | VLM predictor is a stub/optional; local small models can be added later. |
| Scene count grows faster than test coverage | Require a test for every scene before merging a phase. |
| Feasibility of deformable scenes | Start with simple cloth composite in MuJoCo; if unstable, defer to Phase 4.5. |

---

## Decisions to Confirm with User

1. **Project name:** `PIBench` (Physical Intuition Benchmark) — confirm or rename.
2. **First scene choice:** `tower_fall` is proposed. Alternative: `slope_slide` or `push_tip_vs_slide`.
3. **Dashboard tech:** Static HTML generated by Python/Jinja2. OK, or prefer Streamlit/Next.js now?
4. **Venv strategy:** Separate `pibench/.venv` or reuse `chapter01_foundation/.venv`?
5. **Scope of Phase 0:** Should we include a tiny leaderboard page in Phase 0, or defer to Phase 6?
6. **Hardware timeline:** Keep real-robot validation as Phase 7 (document only), or purchase AM-ARM earlier?

---

## Notes

* This plan intentionally keeps Phase 0 small so we can push a working benchmark in one session.
* Every later phase can be paused and resumed while the user continues reading textbook chapters.
* The manifesto concept L is the parent idea; PIBench is the executable first step.
