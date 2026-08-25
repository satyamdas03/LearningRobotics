"""Contact & friction scene suite."""
from pibench.scenes.contact.friction_pile import FrictionPile
from pibench.scenes.contact.peg_in_hole import PegInHole
from pibench.scenes.contact.push_tip_vs_slide import PushTipVsSlide
from pibench.scenes.contact.slip_grip import SlipGrip
from pibench.scenes.contact.stack_overhang import StackOverhang
from pibench.scenes.contact.stack_stability import StackStability
from pibench.scenes.contact.wedge_insert import WedgeInsert

__all__ = [
    "FrictionPile",
    "PegInHole",
    "PushTipVsSlide",
    "SlipGrip",
    "StackOverhang",
    "StackStability",
    "WedgeInsert",
]
