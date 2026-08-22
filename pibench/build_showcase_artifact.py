"""Build a self-contained HTML artifact from the rendered showcase images."""
from __future__ import annotations

import base64
import json
from pathlib import Path


SHOWCASE_DIR = Path(__file__).parent / "output" / "showcase"
OUTPUT_HTML = Path(__file__).parent / "output" / "showcase" / "index.html"


CAPTIONS: dict[str, str] = {
    "arm_default.png": "6-DOF spatial arm in its home configuration. Six revolute joints give the end-effector full 3D position + orientation control.",
    "arm_ik_solution.png": "Chapter 5 numeric IK drives the arm to a reachable target pose using the damped Jacobian pseudoinverse.",
    "arm_dynamics_pose.png": "Chapter 6 — Dynamics pose. The arm's mass matrix, Coriolis forces, and gravity vector govern how it responds to torques.",
    "arm_gravity_comp.png": "Chapter 7 — Gravity compensation. The controller cancels gravity so the arm stays at the commanded configuration without sagging.",
    "arm_pid.png": "Chapter 7 — Joint-space PID. Independent-joint PD plus gravity feedforward drives the arm back to a desired set point.",
    "arm_computed_torque.png": "Chapter 7 — Computed torque. Inverse-dynamics linearization cancels nonlinear dynamics and enforces linear error dynamics.",
    "arm_operational_space.png": "Chapter 7 — Operational space. Resolved-acceleration control tracks an end-effector pose via the Jacobian pseudoinverse.",
    "arm_motion_planning.png": "Chapter 8 — Motion planning. RRT* finds a collision-free joint-space path around static obstacles.",
    "towerfall_seed0.png": "TowerFall — two towers on a tilting platform. The narrow tower is the one that will fall first.",
    "slopeslide_seed0.png": "SlopeSlide — a block on an incline. The answer depends on whether tan(θ) exceeds the static-friction coefficient.",
    "supportbalance_seed0.png": "SupportBalance — a loaded beam. The support must sit at the weighted center of mass for zero net torque.",
    "toppledirection_seed0.png": "ToppleDirection — an off-center tower on a tilted platform. The CoM offset predicts the fall direction.",
    "pendulumswing_seed0.png": "PendulumSwing — two pendulums released together. Period scales with √(length/g), independent of mass.",
    "collisionbounce_seed0.png": "CollisionBounce — a 1D elastic collision. Momentum + kinetic-energy conservation decide the final speeds.",
    "projectilehit_seed0.png": "ProjectileHit — a launched ball. Range is v² sin(2θ)/g, validated against the MuJoCo rollout.",
    "pushtipvsslide_seed0.png": "PushTipVsSlide — a block pushed at different heights. Push high and it tips; push low and it slides.",
    "stackstability_seed0.png": "StackStability — a side tap from a moving ball. Survives if the impact stays inside the stack's support polygon.",
    "wedgeinsert_seed0.png": "WedgeInsert — triangular wedge driven into a gap. Clearance and jamming geometry decide fits vs jams.",
    "frictionpile_seed0.png": "FrictionPile — three objects to push. The hardest to start moving maximizes μ_s × mass.",
    "slipgrip_seed0.png": "SlipGrip — parallel-jaw gripper lifts a block. Lifts only if total friction (2 μ F_grip) can support the weight.",
    "drawerpull_seed0.png": "DrawerPull — a prismatic drawer with static friction. Motor force must overcome friction for the drawer to open.",
    "doorswing_seed0.png": "DoorSwing — a hinged door with frictionloss. Applied torque decides whether it swings open or sticks.",
    "ropetension_seed0.png": "RopeTension — two masses linked by a spatial tendon over a pulley. The heavier side descends.",
    "gearturn_seed0.png": "GearTurn — externally meshed gears. Gear A turns counter-clockwise, so gear B turns clockwise.",
    "chaindrape_seed0.png": "ChainDrape — coarse deformable capsule chain draped over a bar. The free-end height is the numeric answer.",
    "massorder_seed0.png": "MassOrder — three pushed blocks on a frictionless surface. The slowest accelerater is the heaviest (F = m a).",
    "frictionorder_seed0.png": "FrictionOrder — three blocks on a tilting platform. The first to slide has the lowest static-friction coefficient.",
    "counterfactualmass_seed0.png": "CounterfactualMass — a tower on a tilted platform. Doubling the top-block mass can shift the combined CoM outside the base.",
    "counterfactualfriction_seed0.png": "CounterfactualFriction — a block at rest on a ramp. If friction were zero, gravity would make it slide down the incline.",
}


def img_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def title_from_filename(filename: str) -> str:
    base = filename.replace("_seed0.png", "").replace(".png", "").replace("_", " ")
    # Title-case each word.
    return " ".join(word.capitalize() for word in base.split())


def classify_cards(cards: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {
        "Robot Foundations": [],
        "PIBench — Statics": [],
        "PIBench — Dynamics": [],
        "PIBench — Contact & Friction": [],
        "PIBench — Articulated": [],
        "PIBench — Deformable": [],
        "PIBench — Parameter Estimation": [],
    }
    for card in cards:
        fn = card["filename"]
        if "arm" in fn:
            groups["Robot Foundations"].append(card)
        elif fn in {
            "towerfall_seed0.png",
            "slopeslide_seed0.png",
            "supportbalance_seed0.png",
            "toppledirection_seed0.png",
        }:
            groups["PIBench — Statics"].append(card)
        elif fn in {
            "pendulumswing_seed0.png",
            "collisionbounce_seed0.png",
            "projectilehit_seed0.png",
        }:
            groups["PIBench — Dynamics"].append(card)
        elif fn in {
            "pushtipvsslide_seed0.png",
            "stackstability_seed0.png",
            "wedgeinsert_seed0.png",
            "frictionpile_seed0.png",
            "slipgrip_seed0.png",
        }:
            groups["PIBench — Contact & Friction"].append(card)
        elif fn in {
            "drawerpull_seed0.png",
            "doorswing_seed0.png",
            "ropetension_seed0.png",
            "gearturn_seed0.png",
        }:
            groups["PIBench — Articulated"].append(card)
        elif fn == "chaindrape_seed0.png":
            groups["PIBench — Deformable"].append(card)
        elif fn in {
            "massorder_seed0.png",
            "frictionorder_seed0.png",
            "counterfactualmass_seed0.png",
            "counterfactualfriction_seed0.png",
        }:
            groups["PIBench — Parameter Estimation"].append(card)
    return groups


def main() -> None:
    cards = []
    for path in sorted(SHOWCASE_DIR.glob("*.png")):
        fn = path.name
        cards.append(
            {
                "filename": fn,
                "title": title_from_filename(fn),
                "caption": CAPTIONS.get(fn, ""),
                "src": img_to_data_uri(path),
            }
        )

    groups = classify_cards(cards)

    # Build group HTML.
    group_html = ""
    for group_name, group_cards in groups.items():
        if not group_cards:
            continue
        cards_html = ""
        for card in group_cards:
            cards_html += f"""
            <article class="card">
              <div class="card-image">
                <img src="{card['src']}" alt="{card['title']}" loading="lazy" />
              </div>
              <div class="card-body">
                <h3 class="card-title">{card['title']}</h3>
                <p class="card-caption">{card['caption']}</p>
              </div>
            </article>
"""
        group_html += f"""
        <section class="group">
          <h2 class="group-title">{group_name}</h2>
          <div class="grid">{cards_html}</div>
        </section>
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LearningRobotics Showcase</title>
  <style>
    :root {{
      --bg: #f8f9fa;
      --surface: #ffffff;
      --text: #111827;
      --text-muted: #64748b;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --border: #e2e8f0;
      --shadow: 0 1px 3px rgba(0,0,0,0.08);
      --radius: 10px;
      --font-body: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    }}

    @media (prefers-color-scheme: dark) {{
      :root:not([data-theme="light"]) {{
        --bg: #0b1120;
        --surface: #111827;
        --text: #f1f5f9;
        --text-muted: #94a3b8;
        --accent: #60a5fa;
        --accent-soft: #1e3a8a;
        --border: #1e293b;
        --shadow: 0 1px 3px rgba(0,0,0,0.35);
      }}
    }}

    :root[data-theme="dark"] {{
      --bg: #0b1120;
      --surface: #111827;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #60a5fa;
      --accent-soft: #1e3a8a;
      --border: #1e293b;
      --shadow: 0 1px 3px rgba(0,0,0,0.35);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      padding: 0;
      font-family: var(--font-body);
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 48px 24px;
    }}

    header {{
      margin-bottom: 48px;
    }}

    .eyebrow {{
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      background: var(--accent-soft);
      padding: 4px 10px;
      border-radius: 999px;
      margin-bottom: 14px;
    }}

    h1 {{
      font-size: clamp(2rem, 5vw, 3.2rem);
      font-weight: 800;
      letter-spacing: -0.02em;
      line-height: 1.1;
      margin: 0 0 16px;
      text-wrap: balance;
    }}

    .lead {{
      font-size: 1.15rem;
      color: var(--text-muted);
      max-width: 680px;
      margin: 0 0 28px;
    }}

    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 24px;
      margin-top: 8px;
    }}

    .stat {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}

    .stat-value {{
      font-family: var(--font-mono);
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--text);
      line-height: 1;
    }}

    .stat-label {{
      font-size: 0.8rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .group {{
      margin-bottom: 56px;
    }}

    .group-title {{
      font-size: 1.25rem;
      font-weight: 700;
      margin: 0 0 20px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 24px;
    }}

    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: var(--shadow);
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .card:hover {{
      transform: translateY(-3px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }}

    @media (prefers-color-scheme: dark) {{
      :root:not([data-theme="light"]) .card:hover {{
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      }}
    }}

    :root[data-theme="dark"] .card:hover {{
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }}

    .card-image {{
      aspect-ratio: 4 / 3;
      background: #0a0a0a;
      overflow: hidden;
    }}

    .card-image img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}

    .card-body {{
      padding: 16px 18px 18px;
    }}

    .card-title {{
      font-size: 1rem;
      font-weight: 700;
      margin: 0 0 8px;
      line-height: 1.25;
    }}

    .card-caption {{
      font-size: 0.9rem;
      color: var(--text-muted);
      margin: 0;
    }}

    footer {{
      margin-top: 24px;
      padding-top: 24px;
      border-top: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 0.9rem;
    }}

    footer a {{
      color: var(--accent);
      text-decoration: none;
    }}

    footer a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <span class="eyebrow">LearningRobotics</span>
      <h1>What we've built so far</h1>
      <p class="lead">
        A visual tour of the MuJoCo simulations behind Chapters 1–8 and the
        PIBench physical-intuition benchmark (Phases 0–7).
      </p>
      <div class="stats">
        <div class="stat">
          <span class="stat-value">8</span>
          <span class="stat-label">Chapters</span>
        </div>
        <div class="stat">
          <span class="stat-value">6</span>
          <span class="stat-label">PIBench suites</span>
        </div>
        <div class="stat">
          <span class="stat-value">29</span>
          <span class="stat-label">Scenes</span>
        </div>
        <div class="stat">
          <span class="stat-value">63</span>
          <span class="stat-label">Tests passing</span>
        </div>
        <div class="stat">
          <span class="stat-value">100%</span>
          <span class="stat-label">Physics oracle</span>
        </div>
      </div>
    </header>

    <main>
      {group_html}
    </main>

    <footer>
      Rendered with MuJoCo 3.11.0. Source code at
      <a href="https://github.com/satyamdas03/LearningRobotics" target="_blank" rel="noopener">satyamdas03/LearningRobotics</a>.
    </footer>
  </div>
</body>
</html>
"""

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Artifact HTML written to: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
