"""Critic: checks a finished vibrational calculation and decides whether to retry or flag it.

Checks (deterministic, no LLM):

* ``not_converged``: the final optimization stopped before ``fmax_final`` -> retry with
  ``critic.retry_optimizer`` and twice the step budget.
* ``imaginary_modes``: modes below ``-critic.imag_threshold_cm`` -> displace along the most
  negative mode (largest atomic step ``critic.follow_amplitude``) and re-relax.
* ``site_changed``: the adsorbate relaxed away from the requested site (e.g. ontop -> bridge)
  -> flag only; the new site is a property of the model, not a numerical failure.
* ``weakly_bound`` / ``desorbed``: no metal atom within the bonding cutoff -> flag only.

The LangGraph ``critic`` node retries at most ``critic.max_retries`` times, then passes the run on
with its flags; the validation runner records the same flags per row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ir_agent.config import AppConfig
from ir_agent.runtime import get_logger
from ir_agent.sites import classify_site, site_changed

log = get_logger("critic")

RETRY_CODES = ("not_converged", "imaginary_modes")


@dataclass
class Issue:
    code: str
    action: str  #: "retry" | "flag"
    message: str


@dataclass
class Critique:
    issues: list[Issue] = field(default_factory=list)
    n_imag: int = 0
    lowest_cm: float | None = None
    site: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def needs_retry(self) -> bool:
        return any(i.action == "retry" for i in self.issues)

    @property
    def codes(self) -> list[str]:
        return [i.code for i in self.issues]

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok, "needs_retry": self.needs_retry}


def critique(plan, ps, res, cfg: AppConfig) -> Critique:
    """Assess ``res`` (IRResult) for ``ps`` (PreparedSystem) under ``plan``."""
    c = cfg.critic
    out = Critique()
    freqs = np.asarray(res.frequencies_cm, float)
    if freqs.size:
        out.lowest_cm = float(freqs.min())
    out.n_imag = int((freqs < -c.imag_threshold_cm).sum())
    if not ps.converged.get("final", True):
        out.issues.append(Issue("not_converged", "retry",
                                f"final optimization stopped before fmax={cfg.optimization.fmax_final} eV/A "
                                f"({ps.opt_steps.get('final', '?')} steps)"))
    if out.n_imag:
        out.issues.append(Issue("imaginary_modes", "retry",
                                f"{out.n_imag} imaginary mode(s) below -{c.imag_threshold_cm:g} cm^-1 "
                                f"(lowest {out.lowest_cm:.0f} cm^-1)"))
    if plan.surface is not None and ps.kind in ("surface", "cluster"):
        final = ps.stages.get("final", ps.atoms)
        info = classify_site(final, ps.adsorbate_indices, surface_normal=ps.surface_normal)
        out.site = {**info.as_dict(), "requested": plan.surface.site}
        if info.site in ("physisorbed", "desorbed"):
            code = "weakly_bound" if info.site == "physisorbed" else "desorbed"
            out.issues.append(Issue(code, "flag",
                                    f"adsorbate {info.site}: shortest {info.anchor}-metal contact "
                                    f"{info.min_distance:.2f} A > bonding cutoff {info.cutoff:.2f} A"))
        elif site_changed(plan.surface.site, info):
            out.issues.append(Issue("site_changed", "flag",
                                    f"requested {plan.surface.site}, relaxed to {info.label} "
                                    f"({info.anchor} bonded to {info.n_metal} metal atoms)"))
    return out


def follow_imaginary_mode(ps, res, amplitude: float) -> float:
    """Displace ``ps.atoms`` along the most negative mode; returns that mode's wavenumber."""
    from ir_agent.analysis import _cart_modes

    k = int(np.argmin(res.frequencies_cm))
    disp = _cart_modes(res, ps.atoms)[k]  # (len(res.indices), 3)
    step = amplitude * disp / np.abs(disp).max()
    pos = ps.atoms.get_positions()
    pos[res.indices] += step
    ps.atoms.set_positions(pos)
    return float(res.frequencies_cm[k])


def prepare_retry(plan, ps, res, crit: Critique, cfg: AppConfig) -> tuple[AppConfig, list[str]]:
    """Apply the retry strategy for ``crit``; returns (config for the retry, actions taken)."""
    c = cfg.critic
    actions: list[str] = []
    new_cfg = cfg
    if "imaginary_modes" in crit.codes and res.modes_mw is not None:
        nu = follow_imaginary_mode(ps, res, c.follow_amplitude)
        actions.append(f"displaced {c.follow_amplitude:g} A along the {nu:.0f} cm^-1 mode")
    if "not_converged" in crit.codes:
        o = cfg.optimization
        opt = o.model_copy(update={"optimizer": c.retry_optimizer, "max_steps": 2 * o.max_steps,
                                   "stall_steps": 2 * o.stall_steps})
        new_cfg = cfg.model_copy(update={"optimization": opt})
        actions.append(f"re-relax with {c.retry_optimizer}, max_steps {opt.max_steps}")
    log.info("critic retry: %s", "; ".join(actions) or "none")
    return new_cfg, actions
