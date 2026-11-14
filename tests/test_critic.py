"""Site classification, critic checks and the critic retry loop in the LangGraph workflow."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import add_adsorbate, fcc111

from ir_agent.agent.graph import build_graph
from ir_agent.agent.plan import SimulationPlan
from ir_agent.critic import critique
from ir_agent.pipeline import PreparedSystem
from ir_agent.sites import classify_site, site_changed
from ir_agent.validation.run import parse_surface_spec
from ir_agent.vibrations.ir import IRResult


def _co_on_pt(site: str, height: float = 1.85):
    slab = fcc111("Pt", size=(3, 3, 3), vacuum=8.0)
    co = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]])  # C down
    n = len(slab)
    add_adsorbate(slab, co, height, site, mol_index=0)
    return slab, list(range(n, n + 2))


@pytest.mark.parametrize("site,expected,n", [("ontop", "ontop", 1), ("bridge", "bridge", 2),
                                             ("fcc", "fcc", 3), ("hcp", "hcp", 3)])
def test_classify_site_on_pt111(site, expected, n):
    height = {"ontop": 1.85, "bridge": 1.5, "fcc": 1.3, "hcp": 1.3}[site]
    atoms, ads = _co_on_pt(site, height)
    info = classify_site(atoms, ads)
    assert info.anchor == "C"
    assert info.site == expected and info.n_metal == n
    assert not site_changed(site, info)


def test_classify_site_weak_binding_and_hop():
    atoms, ads = _co_on_pt("ontop", 3.2)
    assert classify_site(atoms, ads).site == "physisorbed"
    atoms, ads = _co_on_pt("ontop", 6.0)
    assert classify_site(atoms, ads).site == "desorbed"
    atoms, ads = _co_on_pt("bridge", 1.5)
    assert site_changed("ontop", classify_site(atoms, ads))
    # default adsorbate detection by element
    assert classify_site(atoms).anchor_index == ads[0]


def test_parse_surface_spec():
    assert parse_surface_spec("CO/Pt/111") == ("CO", "Pt", "111", "ontop")
    assert parse_surface_spec("CO/Pt/111/bridge") == ("CO", "Pt", "111", "bridge")
    with pytest.raises(ValueError):
        parse_surface_spec("CO/Pt")


def _ps(atoms, ads, kind="surface", converged=True):
    return PreparedSystem(atoms, kind, ads, stages={"final": atoms.copy()}, converged={"final": converged},
                          opt_steps={"final": 400})


def test_critique_flags(cfg):
    from ir_agent.agent.plan import SurfaceSpec

    atoms, ads = _co_on_pt("bridge", 1.5)
    plan = SimulationPlan(adsorbate="CO", calculator="emt", surface=SurfaceSpec(metal="Pt"))
    res = IRResult(frequencies_cm=np.array([-120.0, 300.0, 2000.0]), intensities_km_mol=None)
    c = critique(plan, _ps(atoms, ads, converged=False), res, cfg)
    assert set(c.codes) == {"not_converged", "imaginary_modes", "site_changed"}
    assert c.needs_retry and c.n_imag == 1 and c.site["site"] == "bridge"
    ok = critique(plan.model_copy(update={"surface": SurfaceSpec(metal="Pt", site="bridge")}),
                  _ps(atoms, ads), IRResult(np.array([300.0, 2000.0]), None), cfg)
    assert ok.ok and not ok.needs_retry


def test_critic_retries_imaginary_mode_then_passes(cfg, monkeypatch):
    """calculation -> critic (imaginary mode: follow it) -> calculation -> critic (ok) -> ..."""
    import ir_agent.pipeline as pipeline

    calls = []

    def fake_run_vibrations(plan, cfg_, ps, run_dir=None):
        calls.append(ps.atoms.get_positions().copy())
        ps.stages["final"] = ps.atoms.copy()
        ps.converged["final"] = True
        modes = np.zeros((6, 2))
        modes[2, 0], modes[5, 0] = 1 / np.sqrt(2), -1 / np.sqrt(2)  # z stretch
        modes[0, 1], modes[3, 1] = 1 / np.sqrt(2), 1 / np.sqrt(2)
        freqs = np.array([-150.0, 2100.0]) if len(calls) == 1 else np.array([60.0, 2100.0])
        return IRResult(freqs, None, modes_mw=modes, indices=[0, 1], route="frequencies-only")

    monkeypatch.setattr(pipeline, "run_vibrations", fake_run_vibrations)
    plan = SimulationPlan(adsorbate="CO", calculator="emt", dipole_method="none")
    final = build_graph().invoke({"config": cfg, "plan": plan, "use_llm": False})
    assert not final.get("error"), final.get("error")
    hist = final["critic_history"]
    assert [h["attempt"] for h in hist] == [0, 1]
    assert hist[0]["needs_retry"] and "imaginary_modes" in [i["code"] for i in hist[0]["issues"]]
    assert hist[0]["actions"] and hist[1]["ok"]
    assert len(calls) == 2 and np.abs(calls[1] - calls[0]).max() == pytest.approx(0.05, abs=1e-6)
    assert "## Critic checks" in final["report_markdown"]


def test_critic_gives_up_after_max_retries(cfg, monkeypatch):
    import ir_agent.pipeline as pipeline

    def always_unconverged(plan, cfg_, ps, run_dir=None):
        ps.stages["final"] = ps.atoms.copy()
        ps.converged["final"] = False
        return IRResult(np.array([2100.0]), None, modes_mw=np.eye(6, 1), indices=[0, 1])

    monkeypatch.setattr(pipeline, "run_vibrations", always_unconverged)
    plan = SimulationPlan(adsorbate="CO", calculator="emt", dipole_method="none")
    final = build_graph().invoke({"config": cfg, "plan": plan, "use_llm": False})
    hist = final["critic_history"]
    assert len(hist) == cfg.critic.max_retries + 1
    assert final["config"].optimization.optimizer == cfg.critic.retry_optimizer
    assert "not_converged" in final["report_markdown"]
