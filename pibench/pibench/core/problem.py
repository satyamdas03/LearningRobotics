"""Abstract problem definition for PIBench."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AnswerType(str, Enum):
    """Supported answer types."""
    CHOICE = "choice"
    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    ORDER = "order"


class Question(BaseModel):
    """A question posed by a PIBench problem."""
    text: str
    answer_type: AnswerType
    choices: Optional[list[str]] = None
    units: Optional[str] = None


class GroundTruth(BaseModel):
    """Ground-truth answer plus diagnostic metadata."""
    answer: Any
    explanation: Optional[str] = None
    latent_params: dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    """A predictor's answer to a problem."""

    answer: Any
    reasoning: Optional[str] = None
    # Optional model confidence in [0, 1]; used for calibration analysis.
    confidence: Optional[float] = None


class Problem(ABC):
    """Base class for every PIBench problem.

    Subclasses must implement:
      - `question()` — return a `Question`
      - `ground_truth()` — return a `GroundTruth`
      - `score(prediction)` — return a float in [0, 1]

    The constructor should build the MuJoCo scene (or any other internal state)
    from the provided deterministic `seed`.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._scene_built = False
        self._build_scene()
        self._scene_built = True

    @abstractmethod
    def _build_scene(self) -> None:
        """Build the internal scene state from ``self.seed``.

        This is where subclasses load/generate MJCF and create MuJoCo
        model/data if needed. It is called once in ``__init__``.
        """
        ...

    @abstractmethod
    def question(self) -> Question:
        """Return the question the predictor must answer."""
        ...

    @abstractmethod
    def ground_truth(self) -> GroundTruth:
        """Return the correct answer plus any diagnostic metadata."""
        ...

    @abstractmethod
    def score(self, prediction: Prediction) -> float:
        """Score a prediction against the ground truth.

        Must return 1.0 for correct, 0.0 for incorrect, and optionally a
        fractional value for partial credit.
        """
        ...

    def concept_tags(self) -> list[str]:
        """Return concept labels for this problem.

        Defaults to the suite name plus the keys of the ground-truth latent
        parameters. Subclasses can override by setting ``_concepts``.
        """
        tags: list[str] = []
        suite = getattr(self, "_suite", None)
        if suite:
            tags.append(suite)
        try:
            gt = self.ground_truth()
            tags.extend(str(k) for k in gt.latent_params.keys())
        except Exception:
            pass
        extra = getattr(self, "_concepts", [])
        tags.extend(str(t) for t in extra)
        return sorted(set(tags))

    def _counterfactual_params(self) -> list[str]:
        """Return the names of attributes that can be overridden for a counterfactual.

        Subclasses may override this to expose a custom list. The default
        implementation returns the keys of the most recent ``latent_params`` dict,
        if available.
        """
        try:
            gt = self.ground_truth()
            return sorted(gt.latent_params.keys())
        except Exception:
            return []

    def run_simulation(self, steps: int, dt_scale: int = 1) -> None:
        """Convenience helper: step the MuJoCo simulation forward.

        Subclasses that keep ``self.model`` and ``self.data`` can call this.
        """
        if not hasattr(self, "model") or not hasattr(self, "data"):
            raise RuntimeError("Problem does not expose self.model / self.data")
        import mujoco
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(seed={self.seed})"
