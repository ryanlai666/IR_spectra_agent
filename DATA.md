# Data, models and parameter sets

This project trains nothing and downloads no training dataset. It uses reference data (hard-coded,
with citations), optional third-party parameter sets, and optional pretrained ML potentials.
Everything downloadable lives under `data/raw/`, `data/checkpoints/` or the package caches and is
gitignored.

## 1. Reference vibrational data (in code: `src/ir_agent/validation/reference.py`)

| Item | Source | License | Retrieved | Verified |
|---|---|---|---|---|
| Gas-phase fundamentals + IR intensities of H2O, CO, NH3, CO2 | NIST CCCBDB "Experimental vibrational frequencies" pages: [H2O](https://cccbdb.nist.gov/exp2x.asp?casno=7732185&charge=0), [CO](https://cccbdb.nist.gov/exp2x.asp?casno=630080&charge=0), [NH3](https://cccbdb.nist.gov/exp2x.asp?casno=7664417&charge=0), [CO2](https://cccbdb.nist.gov/exp2x.asp?casno=124389&charge=0). Frequencies from Shimanouchi, *Tables of Molecular Vibrational Frequencies* (NSRDS-NBS 39, 1972) and Huber & Herzberg (1979); intensities as cited on each page. | US Government work (NIST SRD 101) | 2026-11-12 | yes (values transcribed from the pages) |
| **All measured surface bands** (CO/Pt, NH3/Pt, CO/Au, NH3/Au, CO/Cu, CO/Ru, H/Si(100), NH2+H/Si(100), isolated SiOH) and anharmonic gaps | `data/reference/experimental_surface_ir.csv`, `data/reference/anharmonicity.csv`: value, technique, coverage, temperature, citation, DOI (all checked on Crossref), evidence level | citation only (numbers from the literature) | 2026-11-13 | verified = yes only if read in the primary paper (CO/Cu 2077, SiOH 3747); see REPORT.md §9 |
| Experimental harmonic frequencies: H2O w1/w2/w3 = 3832.2/1648.5/3942.5 cm^-1, CO we = 2169.8 cm^-1 | NIST CCCBDB [experimental harmonic frequencies](https://cccbdb.nist.gov/exptriatomicsx.asp) (H2O); Huber & Herzberg (1979) via CCCBDB (CO) | US Government work (NIST SRD 101) | 2026-11-13 | yes |
| NH3 atop on Pt(111), umbrella mode 1090 cm^-1 (molecular NH3, low coverage, EELS) | J. L. Gland & B. A. Sexton, *J. Catal.* 68, 286 (1981), [doi:10.1016/0021-9517(81)90098-1](https://doi.org/10.1016/0021-9517(81)90098-1) | citation only | 2026-11-13 | bibliographic data yes (Crossref); the **number is not verified** (abstract text in search results only; paper not accessible). |
| CO / NH3 on Au(111) | qualitative trend only (weaker binding than Pt) | - | - | **not verified** |

## 2. DFTB+ Slater-Koster parameter sets (legacy `dftb+` preset, Linux/WSL only)

Not redistributed. Download from <https://dftb.org/parameters/download> and keep them in
`data/raw/dftb_params/` (gitignored).
License: see the terms shown on dftb.org for each set (not re-verified here).

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `3ob-3.1.0.zip` | 19,105,640 | `8ef43c46634ed72799b87312983abe82c2b82c5e326ca692a154b1479106f71d` |
| `auorgap-1.1.0.zip` | 2,737,783 | `de8b196c98132b805ec93e4719416ed484afd56c7c6bde51c1349ff9002e00a6` |
| `mio-1.1.0.zip` | 1,676,953 | `6dea68dd5ebe13a17b1b5e39e37b5e30b6873162db8f45be78ebe9db2a9ee8f6` |

Use: unzip and `export DFTB_PREFIX=/path/to/<set>/` (or `calculators.dftb.skf_dir` in the YAML).

## 3. Meta UMA checkpoint (`uma` presets) - GATED, downloaded manually

| Item | Value |
|---|---|
| Repository | <https://huggingface.co/facebook/UMA> (gating: manual approval; anonymous download returns HTTP 401) |
| License | HF metadata says `license: other`; the repo ships `LICENSE` and `USE_POLICY.md` (gated, so not read here - accept the terms on the HF page; commonly described as the FAIR Chemistry License, **not verified**) |
| Checkpoint used | `uma-s-1p2p1` (UMA-S 1.2.1; `checkpoints/uma-s-1p2p1.pt`, 2,333,393,615 B). fairchem 2.21's registry lists `uma-s-1p2` but not `1p2p1`; the local file loads with `load_predict_unit` (291.5 M parameters, tasks omc/omol/odac/oc20/omat/oc22/oc25). |
| Location | `data/checkpoints/uma-s-1p2p1.pt` (gitignored). Download it manually from the gated repo and place it there; this code never fetches it unless `allow_hub_download: true`. |
| MD5 | `3497615fd30a24c5b35cd3b41a682e6e` - checked by `ir_agent.calculators.uma.verify_checkpoint` before every load and by `scripts/run_uma_run_a.py`; a mismatch is a hard error |
| Loading | `fairchem.core.units.mlip_unit.load_predict_unit(path, device=...)` (fairchem-core 2.21). This skips the gated `references/iso_atom_elem_refs.yaml` (only used for isolated single-atom, non-periodic systems) and `references/form_elem_refs.yaml` (formation energies); IR uses forces only. |
| Citation | Wood et al., "UMA: A Family of Universal Models for Atoms", [arXiv:2506.23971](https://arxiv.org/abs/2506.23971) |

## 4. Orb models (`orb-mol`, `orb-omat` presets) - open weights

| Item | Value |
|---|---|
| Source | <https://github.com/orbital-materials/orb-models> (README + MODELS.md read 2026-11-13), PyPI `orb-models==0.7.0` |
| License | Apache-2.0 (repo LICENSE file and README: "Orb models are licensed under the Apache License, Version 2.0") |
| `orbmol-v2` | OMol25 + OPoly26 (wB97M-V/def2-TZVPD), conservative, needs total charge + spin multiplicity. `orbmol-v2-teqabfhg-20260523.ckpt`, 103,552,643 bytes, SHA-256 `3dbef70b5fdc9124392181b0d8686c79f1c02bf5fefcb878a4eaa51d03e4f5e8` |
| `orb-v3-conservative-inf-omat` | OMat24 (bulk), conservative, effectively unlimited neighbours. `orb-v3-conservative-inf-omat-20250404.ckpt`, 102,097,517 bytes, SHA-256 `0a41ef1132ad9c41ee0c7b9d855fccabfefb97abe04625c97ec0d07930b6c0f0` |
| Download | public S3 `https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/...` (no auth) into `data/raw/orb/` (gitignored) |
| Citation | Rhodes et al., "Orb-v3: atomistic simulation at scale", [arXiv:2504.06231](https://arxiv.org/abs/2504.06231) |

## 5. UPET / PET models (`pet-omol`, `pet-mad` presets) - open weights

| Item | Value |
|---|---|
| Source | <https://huggingface.co/lab-cosmo/upet> (`gated: false`, card metadata `license: bsd-3-clause`, checked 2026-11-13), PyPI `upet==0.3.1` (BSD-3-Clause) |
| `pet-omol-l-v1.0.0.ckpt` | PET-OMol L (OMol25, wB97M-V), reads charge + spin. 1,031,149,827 bytes, SHA-256 `aba2ca84306e9c51abe9172df05fb950507391c5412716c9a81883c320500093` (verified) |
| `pet-omol-s-v1.0.0.ckpt` | PET-OMol S, used for development and the smoke test. 114,495,843 bytes, SHA-256 `9a78baeb288751490bb351371fc4df41505b7165b6f74ef2d61125d28fe3d298` (verified) |
| `pet-mad-s-v1.6.0.ckpt` | PET-MAD-1.6 S (r2SCAN, 102 elements, incl. catalytic surfaces), slab preset. 113,487,103 bytes, SHA-256 `3f52e97c459b2a28a94fc49b8a2f9a92551231a0ea4f5b5db2396f12d8ffa1e9` (verified). |
| Download | `https://huggingface.co/lab-cosmo/upet/resolve/main/models/<file>` (resumable curl) into `data/raw/pet/` (gitignored; `SHA256SUMS` there) |
| Citation | PET-MAD-1.6: [arXiv:2603.02089](https://arxiv.org/abs/2603.02089). Current UPET architecture: Bigi et al., [arXiv:2601.16195](https://arxiv.org/abs/2601.16195) (the upet README lists it for PET-OAM/OMat/OMAD/OMATPES/SPICE; PET-OMol is not named there, so its citation is unverified) |

## 6. Tools fetched as binaries

| Tool | How | Size here |
|---|---|---|
| xtb 6.7.1 (GFN2-xTB executable, LGPL-3.0) | `conda create -p ./.tools/xtb -c conda-forge xtb=6.7.1` (win-64 build) | 684 MB (includes MKL runtime) |
