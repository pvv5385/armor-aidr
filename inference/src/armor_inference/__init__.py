"""`armor-inference` — Armor's optional model-backed inference sidecar.

Armor is fully functional without this service: every check runs its
deterministic path and the product enforces policy. This tier is strictly
additive, and the API reaches it only over HTTP — there is no in-process model
loading anywhere in `armor-api`.
"""

__version__ = "0.1.0"
