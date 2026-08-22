"""Optional LLM predictor using the Anthropic API.

This is the text-only precursor to a full VLM predictor. It sends the problem's
question text to Claude and returns the model's answer. If the Anthropic SDK or
API key is missing, it falls back to random guessing.

Responses are cached locally by (problem class, seed) so repeated evaluations do
not cost API credits.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pibench.core.problem import Prediction, Problem
from pibench.predictors.base import Predictor
from pibench.predictors.random_predictor import RandomPredictor


class LLMPredictor(Predictor):
    """Claude text predictor for PIBench problems.

    Requires ``ANTHROPIC_API_KEY`` environment variable. Without it, the predictor
    falls back to the random baseline.
    """

    name: str = "llm"

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", cache_dir: Path | None = None) -> None:
        self.model = model
        self._random = RandomPredictor(seed=0)
        self._client: Any | None = None

        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "output" / "llm_cache"
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = Anthropic(api_key=api_key)
        except Exception:
            self._client = None

    def _cache_key(self, problem: Problem) -> str:
        raw = f"{problem.__class__.__name__}:{problem.seed}:{self.model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cache(self, key: str) -> dict | None:
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        path = self._cache_dir / f"{key}.json"
        path.write_text(json.dumps(data), encoding="utf-8")

    def _extract_answer(self, text: str, problem: Problem) -> Any:
        """Naive answer extraction: look for a known choice/boolean/numeric."""
        q = problem.question()
        text_lower = text.lower().strip()

        if q.choices:
            for choice in q.choices:
                if choice.lower() in text_lower:
                    return choice
            # If the answer is just one of the choices (common for Claude).
            if text_lower in [c.lower() for c in q.choices]:
                return text_lower

        if q.answer_type.value == "boolean":
            if "yes" in text_lower:
                return "yes"
            if "no" in text_lower:
                return "no"

        if q.answer_type.value == "numeric":
            # Try to pull the first number from the text.
            import re

            match = re.search(r"[-+]?\d*\.?\d+", text)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    pass

        # Fallback: first token.
        return text.strip().split()[0]

    def predict(self, problem: Problem) -> Prediction:
        if self._client is None:
            # API not available: fall back to random but mark reasoning.
            pred = self._random.predict(problem)
            return Prediction(
                answer=pred.answer,
                reasoning="LLM unavailable; random fallback.",
                confidence=pred.confidence,
            )

        key = self._cache_key(problem)
        cached = self._load_cache(key)
        if cached is not None:
            return Prediction(
                answer=cached["answer"],
                reasoning=cached.get("reasoning"),
                confidence=cached.get("confidence", 0.7),
            )

        question = problem.question()
        choices_text = ""
        if question.choices:
            choices_text = " Choices: " + ", ".join(question.choices) + "."
        prompt = (
            "You are answering a physical reasoning benchmark question. "
            "Answer as briefly as possible, using only the requested answer format.\n\n"
            f"Question: {question.text}{choices_text}\n\n"
            "Answer:"
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=64,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
        except Exception as exc:
            pred = self._random.predict(problem)
            return Prediction(
                answer=pred.answer,
                reasoning=f"API error ({exc}); random fallback.",
                confidence=pred.confidence,
            )

        answer = self._extract_answer(raw_text, problem)
        reasoning = raw_text.strip()
        confidence = 0.7
        self._save_cache(key, {"answer": answer, "reasoning": reasoning, "confidence": confidence})
        return Prediction(answer=answer, reasoning=reasoning, confidence=confidence)
