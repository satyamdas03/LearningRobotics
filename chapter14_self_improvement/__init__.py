"""Chapter 14 — Self-Improving Virtual Real-Sim-Real Loop.

This package closes the loop between failure detection, online system
identification, controller/policy retuning, and A/B validation on a virtual
arm.  Everything is simulation-only; when physical hardware is added, only
the ``RealArm`` adapter needs to change.
"""
