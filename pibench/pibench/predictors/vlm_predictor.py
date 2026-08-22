"""Optional vision-language-model predictor for PIBench.

Renders each problem's MuJoCo scene, then asks a multimodal LLM (Claude via the
Anthropic API) to answer from the image and question text. If the Anthropic SDK,
API key, or image rendering stack is unavailable, it falls back to the text-only
LLM predictor (and ultimately to random guessing).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from pibench.core.problem import Prediction, Problem
from pibench.predictors.base import Predictor
from pibench.predictors.llm_predictor import LLMPredictor


class VLMPredictor(Predictor):
    """Claude vision predictor for PIBench problems.

    Requires ``ANTHROPIC_API_KEY`` and either a problem that exposes ``model``
    and ``data`` attributes or a MuJoCo-free problem (in which case it falls back
    to the text-only LLM predictor).
    """

    name: str = "vlm"

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        image_size: tuple[int, int] = (640, 480),
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.image_size = image_size
        self._text_fallback = LLMPredictor(model=model, cache_dir=cache_dir)

        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "output" / "vlm_cache"
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._client: Any | None = None
        self._renderer_available = False
        try:
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = Anthropic(api_key=api_key)
        except Exception:
            self._client = None

        try:
            import mujoco  # noqa: F401
            from PIL import Image  # noqa: F401

            self._renderer_available = True
        except Exception:
            self._renderer_available = False

    def _cache_key(self, problem: Problem) -> str:
        raw = f"{problem.__class__.__name__}:{problem.seed}:{self.model}:{self.image_size}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cache(self, key: str) -> dict | None:
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        path = self._cache_dir / f"{key}.json"
        path.write_text(json.dumps(data), encoding="utf-8")

    def _render_problem(self, problem: Problem) -> bytes | None:
        """Render the problem's scene to a PNG byte string, if possible."""
        if not self._renderer_available:
            return None
        if not hasattr(problem, "model") or not hasattr(problem, "data"):
            return None

        import mujoco
        from PIL import Image

        model = problem.model
        data = problem.data
        try:
            renderer = mujoco.Renderer(model, height=self.image_size[1], width=self.image_size[0])
            renderer.update_scene(data)
            frame = renderer.render()
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    def predict(self, problem: Problem) -> Prediction:
        if self._client is None:
            pred = self._text_fallback.predict(problem)
            return Prediction(
                answer=pred.answer,
                reasoning="VLM unavailable; text-only fallback.",
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

        image_bytes = self._render_problem(problem)
        question = problem.question()
        choices_text = ""
        if question.choices:
            choices_text = " Choices: " + ", ".join(question.choices) + "."

        if image_bytes is None:
            pred = self._text_fallback.predict(problem)
            return Prediction(
                answer=pred.answer,
                reasoning="No renderable scene; text-only fallback. " + (pred.reasoning or ""),
                confidence=pred.confidence,
            )

        prompt_text = (
            "You are answering a physical reasoning benchmark question. "
            "Look at the rendered MuJoCo scene and answer as briefly as possible, "
            "using only the requested answer format.\n\n"
            f"Question: {question.text}{choices_text}\n\n"
            "Answer:"
        )
        encoded = base64.b64encode(image_bytes).decode("utf-8")

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=64,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": encoded,
                                },
                            },
                        ],
                    }
                ],
            )
            raw_text = response.content[0].text
        except Exception as exc:
            pred = self._text_fallback.predict(problem)
            return Prediction(
                answer=pred.answer,
                reasoning=f"Vision API error ({exc}); text fallback.",
                confidence=pred.confidence,
            )

        answer = self._text_fallback._extract_answer(raw_text, problem)
        reasoning = raw_text.strip()
        confidence = 0.7
        self._save_cache(
            key, {"answer": answer, "reasoning": reasoning, "confidence": confidence}
        )
        return Prediction(answer=answer, reasoning=reasoning, confidence=confidence)
