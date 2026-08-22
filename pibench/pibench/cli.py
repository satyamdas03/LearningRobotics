"""PIBench command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pibench.core.registry import list_problems, list_suites
from pibench.core.suite import Suite
from pibench.core.evaluator import Evaluator
from pibench.harness import EvaluationHarness


def _get_predictor(name: str):
    if name == "random":
        from pibench.predictors.random_predictor import RandomPredictor
        return RandomPredictor()
    if name == "physics_oracle":
        from pibench.predictors.physics_oracle import PhysicsOraclePredictor
        return PhysicsOraclePredictor()
    if name == "llm":
        from pibench.predictors.llm_predictor import LLMPredictor
        return LLMPredictor()
    if name == "vlm":
        from pibench.predictors.vlm_predictor import VLMPredictor
        return VLMPredictor()
    raise ValueError(f"Unknown predictor: {name}")


def _available_predictor_choices() -> list[str]:
    choices = ["random", "physics_oracle"]
    try:
        import anthropic  # noqa: F401
        choices.extend(["llm", "vlm"])
    except ImportError:
        pass
    return choices


def cmd_list(args: argparse.Namespace) -> int:
    if args.suites:
        print("Registered suites:")
        for suite in list_suites():
            problems = [cls.__name__ for cls in list_problems(suite)]
            print(f"  {suite}: {problems}")
    else:
        print("Use --suites to list registered suites and scenes.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    predictor = _get_predictor(args.predictor)
    suite_names = args.suites if args.suites else list_suites()
    suites = [Suite(name, seed=args.seed, n_instances=args.n) for name in suite_names]

    from pibench.core.runner import Runner

    runner = Runner(predictor)
    result = runner.run(suites)
    metrics = Evaluator.metrics(result)

    print(f"\nPredictor: {result.predictor}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.1%} "
          f"({metrics['n_correct']}/{metrics['n_total']})")
    print("\nPer-suite accuracy:")
    for sm in metrics["suite_metrics"]:
        print(f"  {sm.suite:20s}: {sm.accuracy:.1%} ({sm.n_correct}/{sm.n_total})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
        print(f"\nDetailed results written to {out_path}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    try:
        import mujoco
        from PIL import Image
    except ImportError as exc:
        print(f"Rendering requires mujoco and pillow: {exc}", file=sys.stderr)
        return 1

    problem_classes = {cls.__name__: cls for cls in list_problems()}
    if args.problem not in problem_classes:
        print(f"Unknown problem '{args.problem}'. Registered problems: {list(problem_classes)}", file=sys.stderr)
        return 1

    cls = problem_classes[args.problem]
    problem = cls(seed=args.seed)
    model = problem.model
    data = problem.data

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data)
    frame = renderer.render()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out_path)
    print(f"Rendered {args.problem} (seed={args.seed}) to {out_path}")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        print(f"Interactive viewing requires mujoco: {exc}", file=sys.stderr)
        return 1

    import time

    problem_classes = {cls.__name__: cls for cls in list_problems()}
    if args.problem not in problem_classes:
        print(f"Unknown problem '{args.problem}'. Registered problems: {list(problem_classes)}", file=sys.stderr)
        return 1

    cls = problem_classes[args.problem]
    problem = cls(seed=args.seed)
    if not hasattr(problem, "model") or not hasattr(problem, "data"):
        print(f"Problem '{args.problem}' does not expose a MuJoCo model/data", file=sys.stderr)
        return 1

    model = problem.model
    data = problem.data

    print(f"Opening MuJoCo viewer for {args.problem} (seed={args.seed})...")
    print("Controls: left-drag to rotate, right-drag to pan, scroll to zoom, Esc to close.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if args.simulate:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep if args.simulate else 1.0 / 60.0)

    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    leaderboard = EvaluationHarness.build_leaderboard(
        output_dir=args.output_dir,
        results_glob="results_*.json",
    )
    json_path, html_path = EvaluationHarness.write_leaderboard(
        leaderboard,
        output_dir=args.output_dir,
    )
    print(f"Leaderboard written to {json_path}")
    if html_path:
        print(f"HTML leaderboard written to {html_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pibench", description="Physical Intuition Benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered suites and scenes")
    list_parser.add_argument("--suites", action="store_true", help="List suites")
    list_parser.set_defaults(func=cmd_list)

    run_parser = subparsers.add_parser("run", help="Run a predictor on suites")
    run_parser.add_argument(
        "--predictor", default="physics_oracle", choices=_available_predictor_choices(),
        help="Predictor to evaluate",
    )
    run_parser.add_argument(
        "--suites", nargs="+", default=None, help="Suite names (default: all)"
    )
    run_parser.add_argument("--n", type=int, default=10, help="Instances per problem")
    run_parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    run_parser.add_argument("--output", default=None, help="Path to write results JSON")
    run_parser.set_defaults(func=cmd_run)

    render_parser = subparsers.add_parser("render", help="Render a scene to an image")
    render_parser.add_argument("problem", help="Problem class name")
    render_parser.add_argument("--seed", type=int, default=0)
    render_parser.add_argument("--output", default="output/render.png")
    render_parser.set_defaults(func=cmd_render)

    view_parser = subparsers.add_parser("view", help="Open a scene in the interactive MuJoCo viewer")
    view_parser.add_argument("problem", help="Problem class name")
    view_parser.add_argument("--seed", type=int, default=0)
    view_parser.add_argument("--simulate", action="store_true", help="Step physics while viewing")
    view_parser.set_defaults(func=cmd_view)

    lb_parser = subparsers.add_parser("leaderboard", help="Build the leaderboard from saved results")
    lb_parser.add_argument(
        "--output-dir", default="output", help="Directory containing results_*.json"
    )
    lb_parser.set_defaults(func=cmd_leaderboard)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
