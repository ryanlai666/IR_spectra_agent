"""Run C semiconductor-process systems: structures, band assignment and bond checks.

Systems (all neutral, closed shell):

* ``H/Si(100)-2x1``: monohydride, one H on each Si of the dimer (HF-last / H-passivated Si).
* ``NH2+H/Si(100)-2x1``: NH3 dissociated on one dimer, NH2 on one Si and H on the other
  (nitridation, SiN ALD nucleation).
* ``SiOH/SiO2``: an isolated silanol, modelled by the cluster HO-Si(O-SiH3)3 (ALD nucleation).

Two models per Si(100) system, picked by the calculator's periodicity, as for the metal systems:

* ``slab`` (periodic presets): Si(100)-2x1 cell (3.84 x 7.68 A, one dimer), 6 Si layers. The bottom
  Si layer is passivated with two H each (dihydride, bulk bond directions, Si-H 1.48 A); the bottom
  two Si layers are fixed, the caps relax.
* ``cluster`` (non-periodic presets): the one-dimer Si9H12 cluster (2 + 4 + 2 + 1 Si). It is cut from
  the same lattice, every broken subsurface bond is capped with H along the bulk bond direction, and
  the caps are fixed.

The silanol cluster is used for every method (in a periodic box for the periodic presets).
The Hessian is partial: adsorbate atoms + the top Si layer (the dimer, or the silanol Si).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.build import diamond100
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import get_layers
from ase.neighborlist import neighbor_list

SI_A = 5.431  #: Si lattice constant (A, experiment, room temperature)
D_SIH = 1.48  #: Si-H cap / adsorbate bond length (A)
D_SIN = 1.74
D_NH = 1.02
D_SIO = 1.63
D_OH = 0.96
SI_BOND_CUT = 2.6  #: Si-Si bond cut-off for the bulk lattice (A)

SYSTEMS = ("H/Si(100)-2x1", "NH2+H/Si(100)-2x1", "SiOH/SiO2")


@dataclass
class SemiSystem:
    name: str
    atoms: Atoms
    model: str  #: "slab" | "cluster"
    adsorbate_indices: list[int]
    hessian_indices: list[int]  #: adsorbate + top layer
    #: (adsorbate atom, substrate atom) bonds that must survive the relaxation
    attachments: list[tuple[int, int]] = field(default_factory=list)
    #: named bond groups for band assignment, e.g. {"Si-H": [(h, si), ...]}
    bonds: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    #: named angles (a, vertex, b), e.g. {"NH2 scissor": [(h1, n, h2)]}
    angles: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)
    surface_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    @property
    def formula(self) -> str:
        return self.atoms.get_chemical_formula()


# ----------------------------------------------------------------------------- Si(100) lattice
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _si100_lattice(nx: int, ny: int, n_layers: int, vacuum: float) -> tuple[Atoms, np.ndarray, str]:
    """Bulk-terminated diamond(100) slab; returns (slab, layer index per atom, dimer axis 'x'|'y')."""
    probe = diamond100("Si", size=(1, 1, 3), a=SI_A, vacuum=vacuum)
    lay, _ = get_layers(probe, (0, 0, 1), tolerance=0.3)
    top = int(np.argmax(lay))
    i, j, D = neighbor_list("ijD", probe, SI_BOND_CUT)
    back = D[i == top]
    # back bonds of a top atom run along one in-plane axis; the dimer forms along the other one
    axis = "y" if np.abs(back[:, 0]).max() > np.abs(back[:, 1]).max() else "x"
    size = (nx, ny, n_layers) if axis == "x" else (ny, nx, n_layers)
    slab = diamond100("Si", size=size, a=SI_A, vacuum=vacuum)
    slab.pbc = (True, True, True)
    layers, _ = get_layers(slab, (0, 0, 1), tolerance=0.3)
    return slab, layers, axis


def _dimer_pairs(slab: Atoms, layers: np.ndarray, axis: str) -> list[tuple[int, int]]:
    """Pair top-layer atoms along the dimer axis (nearest neighbour along that axis, min. image)."""
    top = [int(k) for k in np.where(layers == layers.max())[0]]
    ax = 0 if axis == "x" else 1
    pairs, used = [], set()
    for a in top:
        if a in used:
            continue
        best, bd = None, np.inf
        for b in top:
            if b == a or b in used:
                continue
            d = slab.get_distance(a, b, mic=True, vector=True)
            if abs(d[1 - ax]) < 0.2 and abs(d[ax]) < bd:
                best, bd = b, abs(d[ax])
        if best is None:
            raise ValueError("could not pair the top-layer atoms into dimers")
        pairs.append((a, best))
        used.update((a, best))
    return pairs


def _make_dimers(slab: Atoms, pairs: list[tuple[int, int]], bond: float = 2.40) -> None:
    pos = slab.get_positions()
    for a, b in pairs:
        d = slab.get_distance(a, b, mic=True, vector=True)
        shift = 0.5 * (np.linalg.norm(d) - bond) * _unit(d)
        pos[a] += shift
        pos[b] -= shift
    slab.set_positions(pos)


def dangling_direction(atoms: Atoms, si: int, partner: int, tilt_deg: float = 20.0) -> np.ndarray:
    """Dangling-bond direction of a dimer atom: surface normal tilted away from the partner."""
    away = -atoms.get_distance(si, partner, mic=True, vector=True)
    away[2] = 0.0
    away = _unit(away)
    t = np.radians(tilt_deg)
    return _unit(np.array([0.0, 0.0, 1.0]) * np.cos(t) + away * np.sin(t))


def si100_slab(n_layers: int = 6, vacuum: float = 12.0, n_fixed_layers: int = 2,
               dimers_per_row: int = 1) -> tuple[Atoms, list[tuple[int, int]], list[int]]:
    """Si(100) slab with ``dimers_per_row`` dimers per cell along the row (1: 2x1 cell, 2: 2x2 cell),
    bottom passivated by H.

    Returns (atoms, dimer pairs, top-layer indices). The bottom ``n_fixed_layers`` Si layers are
    fixed; the bottom H caps relax.
    """
    full, layers, axis = _si100_lattice(2, dimers_per_row, n_layers + 1, vacuum)
    i, j, D = neighbor_list("ijD", full, SI_BOND_CUT)
    bottom = set(np.where(layers == layers.min())[0].tolist())
    keep = [k for k in range(len(full)) if k not in bottom]
    caps = []
    for a, b, d in zip(i, j, D, strict=True):
        if a in bottom or b not in bottom:
            continue
        caps.append(full.positions[a] + D_SIH * _unit(d))
    slab = full[keep]
    lay = layers[keep] - 1
    n_si = len(slab)
    slab += Atoms("H" * len(caps), positions=np.array(caps))
    pairs = _dimer_pairs(slab[:n_si], lay, axis)
    _make_dimers(slab, pairs)
    # bottom Si layers fixed; the dihydride caps stay free: at bulk positions two caps that replace
    # the same missing atom are only 1.42 A apart, and they need to cant away from each other
    fixed = [k for k in range(n_si) if lay[k] < n_fixed_layers]
    slab.set_constraint(FixAtoms(indices=fixed))
    top = [k for k in range(n_si) if lay[k] == lay.max()]
    slab.info["n_si"] = n_si
    return slab, pairs, top


def si9h12_cluster(vacuum: float = 8.0) -> tuple[Atoms, tuple[int, int]]:
    """One-dimer Si9H12 cluster (2 + 4 + 2 + 1 Si) cut from the bulk lattice; caps fixed.

    Returns (cluster, (dimer atom a, dimer atom b)); the dimer atoms keep one dangling bond each.
    """
    slab, layers, axis = _si100_lattice(4, 4, 6, vacuum)
    pairs = _dimer_pairs(slab, layers, axis)
    _make_dimers(slab, pairs)
    a0, b0 = pairs[0]
    i, j, D = neighbor_list("ijD", slab, SI_BOND_CUT)
    nbrs: dict[int, list[tuple[int, np.ndarray]]] = {}
    for a, b, d in zip(i, j, D, strict=True):
        nbrs.setdefault(int(a), []).append((int(b), d))
    top = layers.max()
    # unwrap positions relative to a0 by walking the bond graph
    pos = {a0: slab.positions[a0].copy()}
    pos[b0] = pos[a0] + slab.get_distance(a0, b0, mic=True, vector=True)
    sel = [a0, b0]
    for lvl in (top - 1, top - 2, top - 3):
        need = 1 if lvl == top - 1 else 2
        cands: dict[int, list[np.ndarray]] = {}
        for s in sel:
            for n, d in nbrs[s]:
                if layers[n] == lvl and n not in pos:
                    cands.setdefault(n, []).append(pos[s] + d)
        new = [n for n, ps in cands.items() if len(ps) >= need]
        for n in new:
            pos[n] = cands[n][0]
        sel += new
    n_si = len(sel)
    if [int((layers[sel] == top - k).sum()) for k in range(4)] != [2, 4, 2, 1]:
        raise RuntimeError(f"unexpected Si9 cluster layers: {[int((layers[sel] == top - k).sum()) for k in range(4)]}")
    caps = []
    for s in sel:
        for n, d in nbrs[s]:
            if n in pos and n in sel:
                continue
            if s in (a0, b0) and layers[n] == top:
                continue  # neighbouring dimer atoms of the slab: not bonded to this dimer
            caps.append(pos[s] + D_SIH * _unit(d))
    # top atoms in the slab only have 2 back bonds + the dimer bond; their dangling bond stays open
    clu = Atoms("Si" * n_si, positions=np.array([pos[s] for s in sel]))
    clu += Atoms("H" * len(caps), positions=np.array(caps))
    clu.center(vacuum=vacuum)
    clu.pbc = False
    clu.set_constraint(FixAtoms(indices=list(range(n_si, len(clu)))))
    return clu, (0, 1)


# ----------------------------------------------------------------------------- adsorbates
def _add(atoms: Atoms, symbols: str, positions: list[np.ndarray]) -> list[int]:
    n = len(atoms)
    atoms += Atoms(symbols, positions=np.array(positions))
    return list(range(n, len(atoms)))


def _nh2_positions(atoms: Atoms, si: int, partner: int) -> list[np.ndarray]:
    u = dangling_direction(atoms, si, partner)
    n = atoms.positions[si] + D_SIN * u
    # sp3 N: Si-N-H 112 deg, H-N-H ~107 deg (azimuths 120 deg apart); lone pair toward the partner
    p = _unit(np.cross(u, atoms.get_distance(si, partner, mic=True, vector=True)))
    q = _unit(np.cross(p, u))  # points roughly away from the partner
    t = np.radians(180.0 - 112.0)
    hs = [n + D_NH * (np.cos(t) * u + np.sin(t) * (np.cos(phi) * q + np.sin(phi) * p))
          for phi in (np.radians(60.0), np.radians(-60.0))]
    return [n, *hs]


def build(name: str, model: str) -> SemiSystem:
    """Build system ``name`` (one of :data:`SYSTEMS`) as a ``slab`` or ``cluster`` model."""
    if name == "SiOH/SiO2":
        return silanol_cluster(periodic=(model == "slab"))
    extra_h: list[int] = []
    extra_att: list[tuple[int, int]] = []
    if model == "slab" and name == "NH2+H/Si(100)-2x1":
        # 2x2 cell: in a 2x1 cell each NH2 sits 3.84 A from its periodic image (H...H 2.17 A) and the
        # relaxation does not converge. The second dimer carries H/H (monohydride),
        # so NH2 + H is isolated like in the one-dimer cluster; those H are not in the Hessian.
        atoms, pairs, _ = si100_slab(dimers_per_row=2)
        (a, b), (c, d) = pairs[0], pairs[1]
        hc = atoms.positions[c] + D_SIH * dangling_direction(atoms, c, d)
        hd = atoms.positions[d] + D_SIH * dangling_direction(atoms, d, c)
        extra_h = _add(atoms, "HH", [hc, hd])
        extra_att = [(extra_h[0], c), (extra_h[1], d)]
        top = [a, b]
    elif model == "slab":
        atoms, pairs, top = si100_slab()
        a, b = pairs[0]
    elif model == "cluster":
        atoms, (a, b) = si9h12_cluster()
        top = [a, b]
    else:
        raise ValueError(f"model must be 'slab' or 'cluster', not {model!r}")
    normal = np.array([0.0, 0.0, 1.0])
    if name == "H/Si(100)-2x1":
        ha = atoms.positions[a] + D_SIH * dangling_direction(atoms, a, b)
        hb = atoms.positions[b] + D_SIH * dangling_direction(atoms, b, a)
        ads = _add(atoms, "HH", [ha, hb])
        att = [(ads[0], a), (ads[1], b)]
        return SemiSystem(name, atoms, model, ads, top + ads, att, {"Si-H": att}, {}, normal)
    if name == "NH2+H/Si(100)-2x1":
        n, h1, h2 = _nh2_positions(atoms, a, b)
        hb = atoms.positions[b] + D_SIH * dangling_direction(atoms, b, a)
        ads = _add(atoms, "NHHH", [n, h1, h2, hb])
        iN, iH1, iH2, iHb = ads
        att = [(iN, a), (iHb, b), (iH1, iN), (iH2, iN), *extra_att]
        return SemiSystem(name, atoms, model, ads, top + ads, att,
                          {"Si-H": [(iHb, b)], "N-H": [(iH1, iN), (iH2, iN)]},
                          {"NH2 scissor": [(iH1, iN, iH2)]}, normal)
    raise ValueError(f"unknown system {name!r}; choose from {SYSTEMS}")


def silanol_cluster(periodic: bool = False, vacuum: float = 8.0) -> SemiSystem:
    """HO-Si(O-SiH3)3: isolated silanol on a minimal siloxane cage; terminal H fixed."""
    si0 = np.zeros(3)
    o_top = si0 + D_SIO * np.array([0.0, 0.0, 1.0])
    h_top = o_top + D_OH * _unit(np.array([np.sin(np.radians(62.0)), 0.0, np.cos(np.radians(62.0))]))
    syms, pos = ["Si", "O", "H"], [si0, o_top, h_top]
    caps = []
    for k in range(3):
        phi = np.radians(120.0 * k + 60.0)
        down = _unit(np.array([np.sin(np.radians(70.5)) * np.cos(phi),
                               np.sin(np.radians(70.5)) * np.sin(phi), -np.cos(np.radians(70.5))]))
        ob = si0 + D_SIO * down
        # Si-O-Si = 144 deg (amorphous silica): O->Si_t makes 36 deg with the Si0->O direction
        out = np.array([np.cos(phi), np.sin(phi), 0.0])
        out = _unit(out - (out @ down) * down)
        si_t = ob + D_SIO * (np.cos(np.radians(36.0)) * down + np.sin(np.radians(36.0)) * out)
        syms += ["O", "Si"]
        pos += [ob, si_t]
        axis = _unit(si_t - ob)
        perp = _unit(np.cross(axis, [0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.cross(axis, [1.0, 0.0, 0.0]))
        perp2 = np.cross(axis, perp)
        for m in range(3):
            ang = np.radians(120.0 * m)
            d = np.cos(np.radians(70.5)) * axis + np.sin(np.radians(70.5)) * (np.cos(ang) * perp + np.sin(ang) * perp2)
            caps.append(si_t + D_SIH * _unit(d))
    atoms = Atoms(syms, positions=np.array(pos))
    n_core = len(atoms)
    atoms += Atoms("H" * len(caps), positions=np.array(caps))
    atoms.center(vacuum=vacuum)
    atoms.pbc = periodic
    atoms.set_constraint(FixAtoms(indices=list(range(n_core, len(atoms)))))
    si, o, h = 0, 1, 2
    return SemiSystem("SiOH/SiO2", atoms, "slab" if periodic else "cluster", [o, h], [si, o, h],
                      [(o, si), (h, o)], {"O-H": [(h, o)]}, {})


# ----------------------------------------------------------------------------- checks
def bond_ok(atoms: Atoms, a: int, b: int, tol: float = 1.25) -> bool:
    za, zb = atomic_numbers[atoms[a].symbol], atomic_numbers[atoms[b].symbol]
    return atoms.get_distance(a, b, mic=True) < tol * (covalent_radii[za] + covalent_radii[zb])


def broken_attachments(sysm: SemiSystem, atoms: Atoms | None = None) -> list[str]:
    """Attachments (adsorbate-substrate and N-H bonds) that no longer exist after relaxation,
    plus adsorbate H atoms whose nearest heavy atom changed (H hop)."""
    atoms = atoms if atoms is not None else sysm.atoms
    out = []
    for a, b in sysm.attachments:
        if not bond_ok(atoms, a, b):
            out.append(f"{atoms[a].symbol}{a}-{atoms[b].symbol}{b} {atoms.get_distance(a, b, mic=True):.2f} A")
    heavy = [k for k in range(len(atoms)) if atoms[k].symbol != "H"]
    for a, b in sysm.attachments:
        if atoms[a].symbol != "H":
            continue
        d = [atoms.get_distance(a, k, mic=True) for k in heavy]
        nearest = heavy[int(np.argmin(d))]
        if nearest != b:
            out.append(f"H{a} moved from {atoms[b].symbol}{b} to {atoms[nearest].symbol}{nearest}")
    return out


# ----------------------------------------------------------------------------- bands
BAND_WINDOWS = {
    "Si-H": (1700.0, 2500.0),
    "N-H": (2800.0, 3800.0),
    "O-H": (3000.0, 4200.0),
    "NH2 scissor": (1300.0, 1800.0),
}


def _angle(p: np.ndarray, a: int, v: int, b: int) -> float:
    u, w = p[a] - p[v], p[b] - p[v]
    return float(np.arccos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1.0, 1.0)))


def assign_bands(res, atoms: Atoms, sysm: SemiSystem) -> list[dict]:
    """Named bands of ``sysm`` in ``res`` (IRResult on ``atoms``).

    Bond-stretch bands: the modes in the band window with the largest fraction of the motion along
    the bonds of the group; for two bonds the in-phase / out-of-phase combinations are labelled
    'sym' / 'asym'. Bend bands: the mode in the window that changes the angle most per unit motion.
    """
    from ir_agent.analysis import _cart_modes

    if res.modes_mw is None:
        return []
    disp = _cart_modes(res, atoms)  # (n_modes, len(indices), 3)
    where = {int(a): k for k, a in enumerate(res.indices)}
    freqs = np.asarray(res.frequencies_cm, float)
    pos = atoms.get_positions()
    ints = res.intensities_km_mol
    out: list[dict] = []

    def full(k: int) -> np.ndarray:
        d = np.zeros((len(atoms), 3))
        for a, r in where.items():
            d[a] = disp[k][r]
        return d

    for group, pairs in sysm.bonds.items():
        lo, hi = BAND_WINDOWS[group]
        units = [_unit(pos[a] - pos[b]) for a, b in pairs]
        cands = []
        for k in np.where((freqs > lo) & (freqs < hi))[0]:
            d = full(int(k))
            proj = np.array([(d[a] - d[b]) @ u for (a, b), u in zip(pairs, units, strict=True)])
            frac = float(np.sqrt((proj**2).sum()) / (np.linalg.norm(d) + 1e-12))
            cands.append((frac, int(k), proj))
        cands.sort(reverse=True)
        for frac, k, proj in cands[: len(pairs)]:
            if frac < 0.3:
                continue
            label = group + " stretch"
            if len(pairs) == 2:
                label += " (sym)" if proj[0] * proj[1] > 0 else " (asym)"
            out.append({"band": label, "mode": k, "freq_cm": float(freqs[k]), "character": round(frac, 2),
                        "intensity_km_mol": None if ints is None else float(ints[k])})
    for group, triples in sysm.angles.items():
        lo, hi = BAND_WINDOWS[group]
        best = None
        for k in np.where((freqs > lo) & (freqs < hi))[0]:
            d = full(int(k))
            eps = 1e-3 / (np.abs(d).max() + 1e-12)
            dth = sum(abs(_angle(pos + eps * d, *t) - _angle(pos, *t)) for t in triples) / eps
            score = dth / (np.linalg.norm(d) + 1e-12)
            if best is None or score > best[0]:
                best = (score, int(k))
        if best is not None:
            k = best[1]
            out.append({"band": group, "mode": k, "freq_cm": float(freqs[k]), "character": round(best[0], 2),
                        "intensity_km_mol": None if ints is None else float(ints[k])})
    return out
