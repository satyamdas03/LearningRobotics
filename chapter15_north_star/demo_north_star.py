"""Run the Milestone 9 north-star demo end-to-end."""
from __future__ import annotations

import json
from pathlib import Path

from chapter15_north_star.north_star import NorthStarDemo

REPO_ROOT = Path(__file__).parent.parent
SCENE_XML = REPO_ROOT / "chapter10_perception" / "scene.xml"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    demo = NorthStarDemo(
        scene_xml=SCENE_XML,
        reach_tolerance=0.08,
        trajectory_duration=2.5,
    )

    reports = []
    for task_text in [
        "reach the red block",
        "push the red block left of the blue block",
    ]:
        report = demo.run(
            task_text=task_text,
            library_save_path=OUTPUT_DIR / f"north_star_{task_text.replace(' ', '_')}.json",
        )
        reports.append(report.to_dict())
        print("\n".join(report.log))
        print(f"--- arm_reached={report.arm_reached}, skill_saved={report.skill_saved}\n")

    summary_path = OUTPUT_DIR / "north_star_summary.json"
    summary_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
