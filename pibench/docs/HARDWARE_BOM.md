# Hardware BOM Decision Memo — Sub-$500 Robot Stack

> **Context:** We need a real robot platform that can eventually validate the "learn from video, reason with physics, execute safely" north star without blowing the sub-$500 budget. This memo records the decision after surveying low-cost open hardware.

---

## Non-goals for this memo

* We are **not** buying hardware today. Simulation-first remains the rule.
* We are **not** doing a full mechanical review or supply-chain audit.
* We are picking a short list of plausible stacks so the next phase (control / sim-to-real) has a clear default.

---

## Candidate arms compared

| Arm | DOF | Approx. cost | Payload | Reach | Why consider | Why not |
|---|---|---|---|---|---|---|
| **[Forte](https://arxiv.org/html/2507.15693)** | 6 | ~$215 | ~0.5 kg | ~40 cm | Cheapest real 6-DoF arm; sub-mm repeatability claim; good starter validation platform. | Newer, smaller community than AM-ARM; payload modest. |
| **[AM-ARM](https://github.com/liyiteng/AM-ARM)** | 6+1 | ~$380 | ~1 kg | ~52 cm | Established open-source design; 1 kg payload; larger reach; ROS/LeRobot compatible. | More expensive; may exceed tight budget once gripper/compute added. |
| **[U-ARM](https://arxiv.org/html/2509.02437)** | 5 (glove) | ~$50 | N/A | N/A | Ultra-cheap wearable teleop leader for data collection; no robot arm needed. | Not an autonomous arm; only useful for demonstration data. |
| **[Low-Cost Robot Arm](https://github.com/AlexanderKoch-Koch/low_cost_robot)** | 6+6 (leader+follower) | ~$430 | ~0.5 kg | ~35 cm | Ready-made leader-follower pair for imitation learning. | Over budget; more hardware than needed for first validation. |

---

## Recommended configurations

### Option A — "Forte Starter" (best value for first real robot)

| Component | Cost (USD) | Notes |
|---|---|---|
| Forte 6-DoF arm | ~$215 | Main manipulator. |
| USB webcam / phone camera | ~$25 | Visual input for sim-to-real / video-to-skill experiments. |
| Simple parallel-jaw gripper | ~$35 | 3D-printed or low-cost servo gripper. |
| Control board / cables / power | ~$10 | Microcontroller or direct PC control depending on Forte interface. |
| **Total** | **~$285** | Well under $500. |

**Best for:** First contact with real hardware, controller validation, simple pick-place, calibration-loop experiments.

### Option B — "AM-ARM Full Stack" (highest capability within budget)

| Component | Cost (USD) | Notes |
|---|---|---|
| AM-ARM 6+1-DoF arm | ~$380 | Larger reach and payload than Forte. |
| USB webcam | ~$25 | Visual input. |
| Gripper (e.g., Robotis or custom) | ~$50 | Better payload needs stronger gripper. |
| Raspberry Pi 5 (or used PC) | ~$60 | On-robot compute for policy inference. |
| Cables / power / misc | ~$25 | Power supply, USB, mounting. |
| **Total** | **~$540** | Slightly over $500; can trim by reusing existing laptop as compute and skipping Pi. |

**Best for:** Serious manipulation experiments, larger workspace, integration with ROS/LeRobot.

### Option C — "U-ARM Teleop Only" (cheapest data collection)

| Component | Cost (USD) | Notes |
|---|---|---|
| U-ARM glove | ~$50 | Hand-pose teleoperation. |
| Webcam / phone | ~$0 (use existing) | Record demonstrations. |
| Compute | $0 (use laptop) | Process data offline. |
| **Total** | **~$50** | No robot arm yet; pure data-collection tooling. |

**Best for:** Collecting human demonstrations for imitation learning before buying an arm.

---

## Decision

1. **Default purchase target:** Option A — Forte starter stack (~$285). It keeps budget headroom for sensors/gripper upgrades and is the cheapest way to start real-robot validation.
2. **Upgrade path:** If Forte experiments succeed and more payload/reach is needed, sell/donate Forte and move to AM-ARM (Option B).
3. **Data-collection shortcut:** If we want imitation-learning data before buying any arm, add a U-ARM glove (Option C) for ~$50 and map glove poses to the simulated arm.

---

## What this unlocks

| Capability | With Forte starter | With AM-ARM full | With U-ARM only |
|---|---|---|---|
| Sim-to-real policy transfer | ✅ | ✅ | ❌ |
| Real contact-rich manipulation | ⚠️ (small payload) | ✅ | ❌ |
| Teleoperated demonstration data | ⚠️ (need separate leader) | ⚠️ (need separate leader) | ✅ |
| ROS / LeRobot integration | ✅ (community adapters likely) | ✅ | N/A |
| Carry in a backpack | ✅ | ❌ | ✅ |

---

## Open questions before buying

1. Does Forte expose a Python SDK, or only a lower-level protocol? Need to verify before writing sim-to-real bridge.
2. What is the real shipping cost and import duty for the chosen arm to India?
3. Do we need an additional force/torque sensor, or is camera + arm state enough for Phase 7 validation?
4. Should the first real task be a static PIBench-style question (e.g., "which block is heavier?") or a dynamic skill (pick-and-place)?

---

## Links

* Forte paper: <https://arxiv.org/html/2507.15693>
* AM-ARM repo: <https://github.com/liyiteng/AM-ARM>
* U-ARM paper: <https://arxiv.org/html/2509.02437>
* Low-cost robot arm repo: <https://github.com/AlexanderKoch-Koch/low_cost_robot>

---

*Last updated: 2026-08-20*
