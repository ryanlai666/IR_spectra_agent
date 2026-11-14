"""Draw docs/figures/agent_flowchart.png: the agent's LangGraph workflow (no calculations).

Nodes and edges mirror ``src/ir_agent/agent/graph.py``; the sub-labels summarize ``nodes.py``,
``critic.py`` and ``pipeline.py``. README.md carries the same graph in Mermaid. 1600x900 px,
white background, Okabe-Ito colours by role, sized to stay readable at ~400 px wide.

Usage:  python scripts/make_flowchart.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "agent_flowchart.png"

W, H = 1600, 900
INK, INK2, GREY = "#1B1B1B", "#3A3A3A", "#8C8C8C"
#: Okabe-Ito palette, one colour per role
ROLE = {
    "LLM step": "#CC79A7",
    "structure": "#56B4E9",
    "physics compute": "#0072B2",
    "quality check": "#E69F00",
    "output": "#009E73",
}
TITLE_PT, LABEL_PT, SMALL_PT = 28, 17, 16

#: (node name in graph.py, role, sub-label lines, row, column)
NODES = [
    ("planner", "LLM step", ["Gemini → typed plan,", "validated, with fallbacks;", "or JSON plan (--no-llm)"], 0, 0),
    ("geometry", "structure", ["molecule, metal slab", "or cluster model,", "adsorption site, pre-relax"], 0, 1),
    ("calculation", "physics compute", ["tight relax, Hessian by", "finite differences (xtb:", "own --hess), dipoles;", "MSSR on metals"], 0, 2),
    ("critic", "quality check", ["retry: not converged,", "imaginary modes", "flag: site changed,", "weakly bound, desorbed"], 0, 3),
    ("plot_geometry", "structure", ["structure images,", "initial → relaxed"], 1, 0),
    ("spectra", "output", ["broadens frequencies +", "IR intensities into a", "spectrum and plots it"], 1, 1),
    ("analysis", "output", ["vs NIST and measured", "surface bands; report", "(+ optional LLM summary)"], 1, 2),
]
TOOLS = ["UMA", "Orb", "PET", "GFN-xTB"]

X_IN, W_IN, GAP, W_BOX = 25, 175, 40, 305
ROW_Y = [(150, 330), (455, 635)]
CHANNEL_Y = 425


def tint(hex_color: str, keep: float = 0.16) -> tuple[float, float, float]:
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(keep * rgb + (1 - keep) * np.ones(3))


def col_x(c: int) -> tuple[float, float]:
    x0 = X_IN + W_IN + GAP + c * (W_BOX + GAP)
    return x0, x0 + W_BOX


def box(ax, x0, y0, x1, y1, color, lw=3.0, fill=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=14",
                                linewidth=lw, edgecolor=color, facecolor=fill or tint(color), linestyle=ls))


def arrow(ax, p, q, color=INK2, lw=2.6, style="-|>", rad=0.0, ls="-", scale=24):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=scale, linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))


def main() -> Path:
    plt.rcParams["font.family"] = ["Arial", "Segoe UI", "DejaVu Sans"]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    ax.text(X_IN, 50, "IR Spectra Agent: LangGraph workflow", fontsize=30, weight="bold", color=INK, va="center")

    centers = {}
    for name, role, lines, r, c in NODES:
        x0, x1 = col_x(c)
        y0, y1 = ROW_Y[r]
        box(ax, x0, y0, x1, y1, ROLE[role])
        ax.text((x0 + x1) / 2, y0 + 34, name, fontsize=TITLE_PT, weight="bold", color=INK, ha="center", va="center")
        step = 30 if len(lines) <= 3 else 26
        for k, line in enumerate(lines):
            ax.text((x0 + x1) / 2, y0 + 80 + step * k, line, fontsize=LABEL_PT, color=INK2, ha="center", va="center")
        centers[name] = ((x0 + x1) / 2, (y0 + y1) / 2, x0, x1, y0, y1)

    # input (left of row 1)
    yc = sum(ROW_Y[0]) / 2
    ax.text(X_IN + W_IN / 2, yc - 48, "User request", fontsize=21, weight="bold", color=INK, ha="center", va="center")
    ax.text(X_IN + W_IN / 2, yc - 6, "“IR of CO on", fontsize=SMALL_PT, style="italic", color=INK2,
            ha="center", va="center")
    ax.text(X_IN + W_IN / 2, yc + 20, "Pt(111), UMA”", fontsize=SMALL_PT, style="italic", color=INK2,
            ha="center", va="center")
    ax.text(X_IN + W_IN / 2, yc + 58, "or plan.json", fontsize=SMALL_PT, color=INK2, ha="center", va="center")
    arrow(ax, (X_IN + W_IN - 2, yc), (centers["planner"][2], yc))

    # happy path, row 1
    for a, b in [("planner", "geometry"), ("geometry", "calculation"), ("calculation", "critic")]:
        arrow(ax, (centers[a][3], yc), (centers[b][2], yc))

    # tool strip under calculation
    cx, _, x0, x1, _, y1 = centers["calculation"]
    ax.text(x0 - 12, y1 + 38, "tools", fontsize=SMALL_PT, color=INK2, ha="right", va="center", style="italic")
    widths = [68, 58, 58, 106]
    gap = 8
    xs = x0 + (W_BOX - (sum(widths) + gap * (len(widths) - 1))) / 2
    for t, w in zip(TOOLS, widths, strict=True):
        box(ax, xs, y1 + 20, xs + w, y1 + 56, ROLE["physics compute"], lw=1.6, fill="white")
        ax.text(xs + w / 2, y1 + 38, t, fontsize=SMALL_PT, color=INK, ha="center", va="center", weight="bold")
        xs += w + gap
    ax.text(cx, y1 + 76, "MLIPs on GPU; UMA, Orb batched", fontsize=15, color=INK2,
            ha="center", va="center")

    # critic retry loop (curved, above row 1)
    ccx, _, _, _, cy0, _ = centers["critic"]
    kx = centers["calculation"][0]
    arrow(ax, (ccx - 70, cy0 - 2), (kx + 70, cy0 - 2), color=ROLE["quality check"], lw=3.4, rad=0.6, scale=28)
    ax.text(kx + 52, cy0 - 50, "retry, at most critic.max_retries\n(default 1)", fontsize=SMALL_PT,
            color="#8A5A00", ha="right", va="center", weight="bold", linespacing=1.1)

    # critic -> plot_geometry (ok or flagged): down, left along the channel, down
    ccx, _, _, _, _, cy1 = centers["critic"]
    px = centers["plot_geometry"][0]
    ax.plot([ccx, ccx, px], [cy1, CHANNEL_Y, CHANNEL_Y], color=INK2, lw=2.6, solid_capstyle="round",
            solid_joinstyle="round")
    arrow(ax, (px, CHANNEL_Y - 1), (px, ROW_Y[1][0]))
    ax.text(ccx + 14, (cy1 + CHANNEL_Y) / 2 + 4, "ok or\nflagged", fontsize=SMALL_PT, color=INK2, ha="left",
            va="center", linespacing=1.05)

    # happy path, row 2, and the output
    y2 = sum(ROW_Y[1]) / 2
    for a, b in [("plot_geometry", "spectra"), ("spectra", "analysis")]:
        arrow(ax, (centers[a][3], y2), (centers[b][2], y2))
    ox0, ox1 = col_x(3)
    arrow(ax, (centers["analysis"][3], y2), (ox0 + 8, y2))
    xs = np.linspace(ox0 + 30, ox1 - 20, 300)
    nu = np.linspace(0, 1, 300)
    spec = sum(a / (1 + ((nu - c) / 0.022) ** 2) for a, c in [(0.55, 0.18), (1.0, 0.47), (0.35, 0.62), (0.8, 0.86)])
    base, top = ROW_Y[1][0] + 112, ROW_Y[1][0] + 12
    ax.plot([ox0 + 30, ox1 - 20], [base, base], color=GREY, lw=1.5)
    ax.plot(xs, base - (base - top) * spec / spec.max(), color=ROLE["output"], lw=3)
    ax.text((ox0 + ox1) / 2 + 5, base + 34, "IR spectrum + report", fontsize=21, weight="bold", color=INK,
            ha="center", va="center")
    ax.text((ox0 + ox1) / 2 + 5, base + 66, "END", fontsize=SMALL_PT, color=INK2, ha="center", va="center")

    # failure branch (muted grey lane)
    ly0, ly1 = 680, 770
    box(ax, X_IN, ly0, W - 25, ly1, GREY, lw=1.6, fill="#F5F5F5", ls=(0, (6, 4)))
    lyc = (ly0 + ly1) / 2
    ax.text(X_IN + 22, lyc, "error in any node", fontsize=LABEL_PT, color=INK2, ha="left", va="center")
    fx0, fx1 = col_x(1)
    arrow(ax, (X_IN + 235, lyc), (fx0, lyc), color=GREY, ls=(0, (5, 4)))
    box(ax, fx0, ly0 + 12, fx1, ly1 - 12, GREY, lw=2.4, fill="#E9E9E9")
    ax.text((fx0 + fx1) / 2, lyc, "failure", fontsize=TITLE_PT - 4, weight="bold", color="#4A4A4A",
            ha="center", va="center")
    arrow(ax, (fx1, lyc), (fx1 + GAP + 10, lyc), color=GREY)
    ax.text(fx1 + GAP + 22, lyc, "writes an error report → END", fontsize=LABEL_PT, color=INK2, ha="left",
            va="center")

    # role legend
    lx = X_IN
    renderer = fig.canvas.get_renderer()
    for role, color in ROLE.items():
        box(ax, lx, 818, lx + 30, 848, color, lw=2.4)
        t = ax.text(lx + 40, 833, role, fontsize=LABEL_PT + 1, color=INK, ha="left", va="center")
        lx += 40 + t.get_window_extent(renderer).width + 48

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100, facecolor="white")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(f"wrote {main()}")
