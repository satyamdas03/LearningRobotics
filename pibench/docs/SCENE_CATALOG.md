# PIBench Scene Catalog

A living index of every scene in the benchmark, its suite, the physical concepts it tests, and how to run it.

---

## Statics Suite

### `TowerFall`

| Field | Value |
|---|---|
| **File** | `pibench/scenes/statics/tower_fall.py` |
| **Question** | Two towers stand on a tilting platform. Which tower falls? |
| **Answer type** | Choice: `"A"` or `"B"` |
| **Concepts** | Center of mass, support polygon, stability |
| **Textbook link** | Chapter 2 — rigid-body motions, CoM |
| **Run CLI** | `python -m pibench run --suite statics --predictor physics_oracle --n 5` |
| **Render CLI** | `python -m pibench render TowerFall --seed 0 --output output/tower_fall_seed0.png` |

**How ground truth is computed:**  
The platform tilts to a random angle. MuJoCo rolls the simulation forward until the towers settle or fall. A tower is counted as "fallen" if any of its blocks drops below the platform surface by more than a threshold. The label is the tower that loses more blocks / stability.

**Latent parameters logged:** `tower_a_height`, `tower_b_height`, `base_a`, `base_b`, `mass_a`, `mass_b`, `tilt_angle`.

---

## Dynamics Suite

*Coming in Phase 2.*

## Contact Suite

*Coming in Phase 3.*

## Articulated / Deformable Suite

*Coming in Phase 4.*

## Parameter Estimation / Counterfactual Suite

*Coming in Phase 5.*

---

## Concept coverage map

| Concept | TowerFall | SlopeSlide | SupportBalance | CollisionBounce | PendulumSwing | ... |
|---|---|---|---|---|---|---|
| Center of mass | ✅ | ✅ | ✅ | | | |
| Support polygon | ✅ | | ✅ | | | |
| Friction angle | | ✅ | | | | |
| Conservation of momentum | | | | ✅ | | |
| Energy / length scaling | | | | | ✅ | |

*(Map will be filled as phases land.)*
