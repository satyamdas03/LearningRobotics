# Contributing to PIBench

Thanks for helping make physical-intuition evaluation better. This guide covers how to add a new scene, predictor, or metric.

---

## Adding a new scene

1. **Choose a suite** (or propose a new one) aligned with the current textbook chapter or a real-world blocker.
2. **Create a file** under `pibench/scenes/<suite>/<your_scene>.py`.
3. **Inherit from `Problem`** and decorate with `@register_problem("<suite>")`.
4. **Implement the four required methods:**
   * `_build_scene()` — deterministic construction from `self.seed`.
   * `question()` — returns a `Question` with `text`, `answer_type`, and any choices/units.
   * `ground_truth()` — returns a `GroundTruth` with `answer`, `explanation`, and `latent_params`.
   * `score(prediction)` — returns `1.0` for correct, `0.0` for incorrect, or partial credit.
5. **Add a test** in `tests/` that checks the question, ground truth, and scoring.
6. **Document it** in `docs/SCENE_CATALOG.md`.

### Scene design checklist

- [ ] Deterministic for a given seed (use `np.random.default_rng(self.seed)`).
- [ ] Question is unambiguous and answerable from the rendered scene or described state.
- [ ] Ground truth uses MuJoCo rollout or an analytic derivation.
- [ ] Latent parameters are recorded for diagnostic analysis.
- [ ] Score is in `[0, 1]`.
- [ ] No hard-coded paths or OS-specific assumptions.
- [ ] Rendering works (or is gracefully skipped) in headless mode.

---

## Adding a new predictor

1. Create a file under `pibench/predictors/`.
2. Inherit from `Predictor`.
3. Implement `predict(self, problem: Problem) -> Prediction`.
4. Register it in `pibench/cli.py` under `_get_predictor` if you want CLI access.
5. Add a test in `tests/`.

---

## Adding a metric

1. Add the computation to `pibench/core/evaluator.py`.
2. Expose it through `Evaluator.metrics()`.
3. Update the CLI output and dashboard generator if needed.

---

## Code style

* Python 3.11+ type hints.
* `from __future__ import annotations` in every module.
* Pydantic models for config, questions, and results.
* `pytest` for all tests; run `python -m pytest tests -q` before opening a PR.

---

## Commit message convention

```
robotics: add {suite}/{scene} scene
pibench: implement {predictor} predictor
pibench: add {metric} metric
docs: update scene catalog and README
```

---

## Questions?

Open an issue in the main repo or update this document as the project evolves.
