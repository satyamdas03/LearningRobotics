# Super Master Prompt for RoboCAD — Phase 15: GEDA Bridge Integration

> **When to use this prompt:** Only after RoboCAD Phase 14 is complete, tested, and verified end-to-end. Until then, this document stays frozen and the LearningRobotics project works on standalone tasks.

---

## 1. Strategic context

You are the RoboCAD session/model. Your sister project is **LearningRobotics**, a public MuJoCo-based robotics learning journal with 187 passing tests, a physics verifier, a skill library, a vision-learning pipeline, and a physical-intuition benchmark (PIBench).

After deep market research, we chose the revolutionary direction for the combined system:

- **#1 market-value idea:** **RoboCompiler** — human video → physics-grounded, verified robot skill.
  - Adjacent TAM 2026: robot skill-learning platforms **$4.2–5.4 B**, robot learning from demonstration **$3.2–4.1 B**, imitation learning **$2.1–2.8 B**.
  - It attacks data scarcity: real robot episodes cost **$4–30 each**, but human videos are free and abundant. The goal is to compile ordinary RGB videos into transferable, MuJoCo-verified robot skills.

- **#2 market-value idea:** **PhysHack** — LLM writes controller code + MuJoCo self-debug verifier.
  - Adjacent TAM 2026: AI copilots for robot programming **$2.75 B**; AI PLC/industrial-copilot code generation growing at **50 % CAGR to $25.95 B by 2036**.
  - This is the practical developer-tool play, but a narrower moat than RoboCompiler.

**Decision:** GEDA Bridge is **not** the revolution. It is the **delivery infrastructure** for RoboCompiler. Your job in Phase 15 is to make RoboCAD output consumable by LearningRobotics so that RoboCompiler can:
1. Take a human video of a manipulation task.
2. Optionally ask RoboCAD to design a custom end-effector/part if the task requires it.
3. Export that part to MuJoCo with correct inertial properties.
4. Compose it into a task scene in LearningRobotics.
5. Learn/verify the skill in simulation.
6. Package the design + skill + sim + report into a reusable bundle.

---

## 2. What you (RoboCAD) must provide

### 2.1 Stable export API — `ai_cad.geda_bridge.exporter`

Function signature to preserve:

```python
from ai_cad.geda_bridge.exporter import export_to_mujoco, InertialProperties, MjcfExportResult

def export_to_mujoco(
    build123d_part: Any,
    design_id: str,
    output_dir: Path,
    density_kg_m3: float = 7850.0,
    material_name: str = "steel",
    decimate_ratio: float | None = None,
) -> MjcfExportResult:
    ...
```

Required outputs for every design:
- `part.stl` — visual mesh, metric units, Y-up or Z-up consistent with MuJoCo.
- `part_collision.stl` — convex/decomposed collision mesh (decimated if specified).
- `part.mjcf` — MuJoCo MJCF containing `<mesh>`, `<body>`, `<geom>`, `<inertial fullinertia="...">` with mass, CoM, and principal inertia computed from the actual density.
- `part.urdf/` directory or `part.urdf` — optional URDF for ROS2/Gazebo portability.
- `inertial.json` — `InertialProperties` serialized: mass (kg), center_of_mass (m), principal_inertia (kg·m²), density, material.
- `manifest.json` — schema version, design_id, timestamp, source prompt.

**Critical invariants:**
- Units must be **meters, kilograms, seconds**. Convert from mm at export time.
- Inertia must be scaled by the actual density, not trimesh’s default 1.0 kg/m³.
- Output must be deterministic: same prompt + seed → same geometry + inertial properties.
- No hardcoded secrets. No cloud execution of arbitrary generated code.

### 2.2 Assembly-aware export — assemblies from Phase 11

If the design has assemblies/mates, export must expose:
- A top-level `assembly.mjcf` with child bodies connected by joints.
- Joint types, limits, and initial positions preserved from the CAD mate constraints.
- Per-part inertial properties for each component.
- A `BOM.json` listing parts, materials, and quantities.

### 2.3 DFM + tolerances report — from Phase 12

For every exported design, produce `report/manufacturing.json` with:
- manufacturability_score (0.0–1.0)
- estimated_cost_usd
- suggested_process (e.g., FDM, SLA, CNC, sheet-metal)
- tolerance_analysis summary
- critical_dimensions
- material_recommendation
- failure warnings

### 2.4 Packaging endpoint — from Phase 14

A single directory output per design, following **Bundle Schema v2**:

```
{design_id}/
  manifest.json                 # schema_version: "2.0.0"
  design/
    prompt.txt                  # natural-language prompt that generated the part
    part.py                     # build123d source (sandboxed, deterministic)
    part.step
    part.stl
    part_collision.stl
    part.mjcf
    part.urdf                   # or urdf/ directory
    inertial.json
    BOM.json                    # if assembly
  report/
    manufacturing.json          # DFM/cost/tolerances from Phase 12
  scene/
    task.yaml                   # placeholder, to be filled by LearningRobotics
    scene.xml                   # placeholder, to be filled by LearningRobotics
  skill/
    policy.pkl                  # placeholder, to be filled by LearningRobotics
    controller.py               # placeholder, to be filled by LearningRobotics
    eval_metrics.json           # placeholder, to be filled by LearningRobotics
    demo_video.mp4              # placeholder, to be filled by LearningRobotics
```

### 2.5 CLI / API entry points

CLI:
```bash
python -m ai_cad.geda_bridge --prompt "a small wedge for pushing 1kg blocks" \
  --task push --output ./bundles/
```

Backend API (extend `web/backend/main.py`):
- `POST /designs/{id}/simulate` — trigger MuJoCo load check.
- `GET  /designs/{id}/simulation` — get load/success status.
- `GET  /designs/{id}/bundle` — download the bundle zip.
- `POST /capabilities` — register a verified capability (to be called by LearningRobotics after verification).

---

## 3. Interface contract with LearningRobotics

LearningRobotics will consume your bundle and run these steps:

1. **Load MJCF/URDF** into a MuJoCo `mjb` or `mjModel`.
2. **Verify inertial properties** are physically plausible (mass > 0, positive definite inertia, CoM inside bounding box).
3. **Compose scene** by substituting the generated part into the manipulation scene.
4. **Parse the task** from the prompt or from a human video.
5. **Run RoboCompiler** to convert video/objects/relations into a `SkillInstance`.
6. **Verify** the skill with the Chapter 12 physics verifier.
7. **Write back** `skill/`, `scene/`, and verification reports into the bundle.
8. **Call back** to RoboCAD `/capabilities` to register the verified design+skill pair.

Your job is to make steps 1–3 effortless. LearningRobotics owns steps 4–8.

### 3.1 Do not touch LearningRobotics code

- You do **not** modify files in `C:\Users\point\projects\LearningRobotics`.
- You expose a clean API and file schema. The other model/session in LearningRobotics will call it.
- If a change is needed on the LearningRobotics side, document it and request it; do not implement it unilaterally.

---

## 4. Verification thresholds

For a design to be accepted by the GEDA Bridge:

| Metric | Threshold | Who checks |
|---|---|---|
| MJCF loads in MuJoCo without error | pass | RoboCAD `simulate` endpoint |
| Mass > 0 | yes | RoboCAD exporter + LearningRobotics sanity check |
| Inertia positive definite | yes | LearningRobotics physics verifier |
| CoM inside bounding box | yes | LearningRobotics physics verifier |
| manufacturability_score >= 0.7 | yes | RoboCAD DFM report |
| sim_stability >= 0.95 | yes | LearningRobotics 100-step stability rollout |
| success_rate >= 0.8 | yes | LearningRobotics skill verifier |

---

## 5. Minimum viable Phase 15 scope

Do not try to build everything at once. Phase 15 has two sub-phases:

### Phase 15A — RoboCAD-only readiness (1–2 weeks)
- [ ] All Phase 14 exports (STL, STEP, MJCF, URDF, inertial.json, manufacturing.json) are generated deterministically.
- [ ] Exporter tests cover: cube, cylinder, L-bracket, assembly with 2 mates.
- [ ] `/designs/{id}/simulate` endpoint returns load success/failure + any MuJoCo warnings.
- [ ] Bundle packager produces schema v2 directories.
- [ ] 3 example designs committed in `bundles/examples/`:
  - `gripper_cube_grasp`
  - `bracket_hook_hang`
  - `wedge_push_block`

### Phase 15B — Cross-repo handshake (1 week)
- [ ] LearningRobotics reads a RoboCAD bundle and loads the MJCF successfully.
- [ ] LearningRobotics runs a stability rollout; no divergence/explosion.
- [ ] LearningRobotics composes the part into the standard manipulation scene.
- [ ] LearningRobotics verifies one simple skill (e.g., push block) using the generated wedge.
- [ ] Verified bundle is written back and registered via `/capabilities`.

---

## 6. Non-goals for Phase 15

- Do not build a full RL training pipeline in RoboCAD.
- Do not build video parsing in RoboCAD.
- Do not build a marketplace backend in RoboCAD.
- Do not modify LearningRobotics files.
- Do not connect to physical hardware from RoboCAD.

---

## 7. Handoff checklist

Before telling the LearningRobotics session "we are ready," confirm:

- [ ] Phase 14 test suite passes.
- [ ] `pytest ai_cad/geda_bridge/` passes (all exporter/composer/packager tests).
- [ ] 3 example bundles exist and are loadable by a standalone MuJoCo script in LearningRobotics.
- [ ] `README.md` and `PLAN.md` in RoboCAD document the Phase 15 API and bundle schema.
- [ ] This prompt has been shared with the LearningRobotics session.
