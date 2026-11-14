"""Wire the nodes into a LangGraph StateGraph (happy path, critic retry loop, failure branch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from ir_agent.agent.nodes import (
    SimulationState,
    analysis_node,
    calculation_node,
    critic_node,
    failure_node,
    geometry_node,
    planner_node,
    plot_geometry_node,
    spectra_node,
)

ORDER = ["planner", "geometry", "calculation", "critic", "plot_geometry", "spectra", "analysis"]
_NODES = {
    "planner": planner_node,
    "geometry": geometry_node,
    "calculation": calculation_node,
    "critic": critic_node,
    "plot_geometry": plot_geometry_node,
    "spectra": spectra_node,
    "analysis": analysis_node,
}


def _router(next_node: str):
    def route(state: SimulationState) -> str:
        return "failure" if state.get("error") else next_node

    return route


def _critic_router(state: SimulationState) -> str:
    if state.get("error"):
        return "failure"
    return "calculation" if state.get("retry") else "plot_geometry"


def build_graph() -> Any:
    """Compile the workflow. Invoke with ``{"task": ..., "config": AppConfig, ...}``."""
    g = StateGraph(SimulationState)
    for name, fn in _NODES.items():
        g.add_node(name, fn)
    g.add_node("failure", failure_node)
    g.set_entry_point("planner")
    for a, b in zip(ORDER, ORDER[1:], strict=False):
        if a == "critic":
            continue
        g.add_conditional_edges(a, _router(b), {b: b, "failure": "failure"})
    g.add_conditional_edges("critic", _critic_router,
                            {"calculation": "calculation", "plot_geometry": "plot_geometry",
                             "failure": "failure"})
    g.add_conditional_edges("analysis", _router(END), {END: END, "failure": "failure"})
    g.add_edge("failure", END)
    return g.compile()


def save_graph_png(app: Any, path: Path) -> Path | None:
    """Mermaid PNG of the graph (needs network for mermaid.ink); returns None on failure."""
    try:
        png = app.get_graph().draw_mermaid_png()
    except Exception:
        return None
    path.write_bytes(png)
    return path


def mermaid(app: Any) -> str:
    return app.get_graph().draw_mermaid()
