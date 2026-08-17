"""Contact & friction scene suite."""
from pibench.scenes.contact.friction_pile import FrictionPile
from pibench.scenes.contact.push_tip_vs_slide import PushTipVsSlide
from pibench.scenes.contact.slip_grip import SlipGrip
from pibench.scenes.contact.stack_stability import StackStability
from pibench.scenes.contact.wedge_insert import WedgeInsert

__all__ = [
    "FrictionPile",
    "PushTipVsSlide",
    "SlipGrip",
    "StackStability",
    "WedgeInsert",
]
