"""Data models for real-robot validation tasks."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ValidationTask(BaseModel):
    """A single real-robot validation episode.

    The harness executes the action on a real (or mocked) arm, observes the
    physical outcome, and compares it to the predicted answer.
    """

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(..., description="Unique task identifier")
    description: str = Field(default="", description="Human-readable task summary")
    predicted_answer: str = Field(..., description="Answer predicted by the model under test")
    action_type: str = Field(default="reach_q", description="Action primitive: 'reach_q' for now")
    action_params: dict = Field(default_factory=dict, description="Action-specific parameters")
    outcome_key: str = Field(default="reached", description="Key used to label the observed outcome")


class ValidationResult(BaseModel):
    """Result of running one validation task."""

    task_id: str
    predicted: str
    actual: str
    match: bool
    log: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
