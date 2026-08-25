"""Sanity tests for PIBench scene problems."""
from __future__ import annotations

import pytest

from pibench.core.problem import AnswerType, Prediction
from pibench.core.registry import list_problems
from pibench.scenes.contact.peg_in_hole import PegInHole
from pibench.scenes.contact.stack_overhang import StackOverhang
from pibench.scenes.deformable.rope_sag import RopeSag
from pibench.scenes.statics.hanging_beam import HangingBeam


def test_new_problems_are_registered():
    names = {cls.__name__ for cls in list_problems()}
    assert "StackOverhang" in names
    assert "HangingBeam" in names
    assert "PegInHole" in names
    assert "RopeSag" in names


@pytest.mark.parametrize("seed", range(3))
def test_stack_overhang_runs(seed: int):
    p = StackOverhang(seed=seed)
    q = p.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices or []) == {"A", "B", "both", "neither"}
    gt = p.ground_truth()
    assert gt.answer in (q.choices or [])
    assert p.score(Prediction(answer=gt.answer)) == 1.0
    assert p.score(Prediction(answer="wrong")) == 0.0


@pytest.mark.parametrize("seed", range(3))
def test_hanging_beam_runs(seed: int):
    p = HangingBeam(seed=seed)
    q = p.question()
    assert q.answer_type == AnswerType.BOOLEAN
    gt = p.ground_truth()
    assert gt.answer in ("yes", "no")
    assert p.score(Prediction(answer=gt.answer)) == 1.0
    assert p.score(Prediction(answer="wrong")) == 0.0


@pytest.mark.parametrize("seed", range(3))
def test_peg_in_hole_runs(seed: int):
    p = PegInHole(seed=seed)
    q = p.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices or []) == {"fits", "jams"}
    gt = p.ground_truth()
    assert gt.answer in (q.choices or [])
    assert p.score(Prediction(answer=gt.answer)) == 1.0
    assert p.score(Prediction(answer="wrong")) == 0.0


@pytest.mark.parametrize("seed", range(3))
def test_rope_sag_runs(seed: int):
    p = RopeSag(seed=seed)
    q = p.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices or []) == {"A", "B", "same"}
    gt = p.ground_truth()
    assert gt.answer in (q.choices or [])
    assert p.score(Prediction(answer=gt.answer)) == 1.0
    assert p.score(Prediction(answer="wrong")) == 0.0
