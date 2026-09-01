# Physicochemical descriptors for explainable mutagenicity prediction

Code and results for *"From black boxes to chemical interpretability: explainable
deep learning for mutagenicity prediction"* (manuscript in preparation).

A pre-trained SMILES transformer (MolEncoder) is fine-tuned for Ames mutagenicity
prediction with per-atom and molecule-level physicochemical descriptors fused into
the token representation. The central question is whether adding descriptors that
carry explicit chemistry improves both predictive accuracy and the chemical
plausibility of post-hoc attributions.

## Main result

All descriptor sets that are attached **per token** improve ROC-AUC over the
sequence-only baseline. Semi-empirical (GFN2-xTB) descriptors and cheap empirical
ones (Gasteiger charges) perform equivalently.

| Run | ROC-AUC | Δ vs baseline | folds improved |
|---|---|---|---|
| `baseline` | 0.8572 ± 0.0102 | — | — |
| `rdkit_mr` | 0.8649 ± 0.0175 | +0.0077 | 4/5 |
| `xtb_polar` | 0.8663 ± 0.0180 | +0.0091 | 4/5 |
| `pf_gast_neigh_mr` | 0.8696 ± 0.0116 | +0.0123 | 5/5 |
| `gast_neigh` | 0.8701 ± 0.0120 | +0.0128 | 4/5 |
| `orbital_tok` | 0.8702 ± 0.0074 | +0.0129 | 5/5 |
| `neighbors` | 0.8704 ± 0.0117 | +0.0132 | 5/5 |
| `xtb_charge` | 0.8739 ± 0.0098 | +0.0167 | 5/5 |
| `physchem_tok` | 0.8744 ± 0.0160 | +0.0172 | 5/5 |
| `gast_neigh_mr` | 0.8746 ± 0.0120 | +0.0174 | 5/5 |
| `gasteiger` | 0.8746 ± 0.0110 | +0.0174 | 5/5 |
| `pf_gasteiger` | 0.8751 ± 0.0110 | +0.0178 | 5/5 |
| `pf_gast_neigh` | 0.8760 ± 0.0092 | +0.0187 | 5/5 |

Five-fold stratified cross-validation, 7,251 compounds, seed 42. Full statistics,
including paired t-tests across folds, are in [RESULTS.md](RESULTS.md).

Runs prefixed `pf_` use the per-feature attention variant (`run_perfeat.py`); see
[RESULTS.md](RESULTS.md) for why it exists.

## Repository layout

```
data/
  ames_7251.csv          Ames dataset (TDC), filtered to compounds with xTB output
  xtb_7251/              GFN2-xTB outputs — see data/xtb_7251/README.md
models/                  MolEncoder weights (downloaded on first run, not in repo)
runs_7251/               cross-validation results, one directory per descriptor set
  _molecule_branch/      superseded molecule-level-branch runs (metrics only)
scripts/                 reproduction and analysis scripts
src/downstream/toxicity/ fine-tuning, descriptor construction, XAI tooling
```

Each run directory contains `cross_validation_summary.json`,
`cross_validation_summary.csv`, `cross_validation_fold_metrics.csv` and per-fold
subdirectories with checkpoints.

## Data availability

The Ames labels come from the Therapeutics Data Commons. The GFN2-xTB output files
(7,251 single-point calculations, produced with ULYSSES) are archived separately on
Zenodo: **DOI to be added on publication**. A copy is included under `data/xtb_7251/`
for direct reproduction; see [data/xtb_7251/README.md](data/xtb_7251/README.md) for
the file format and the compound-to-calculation mapping.

Four of the 7,255 TDC compounds were excluded because GFN2-xTB did not produce a
usable result (hypervalent or non-convergent species). They are listed in
`data/xtb_7251/ames_out_assignment.csv`.

## Reproducing Table 1

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .

# verify the descriptor data before training
python scripts/verify_7251.py

# run all descriptor configurations
bash scripts/run_table_7251.sh

# per-feature attention variant
python run_perfeat.py
```

`scripts/setup_vm.sh` provisions a fresh Linux VM. Training is CPU-feasible: the
full table took roughly a day on 2 vCPUs.

## Method summary

Descriptors are attached at the **token** level: each atom's value is written onto
the SMILES character(s) representing that atom, standardised using dataset-level
statistics, and projected into the encoder hidden space. Molecule-level descriptors
(frontier-orbital energies, RDKit physicochemical properties) are broadcast to every
token rather than entering through a separate branch.

This last point matters. Earlier runs that routed molecule-level descriptors through
a separate fusion branch *reduced* accuracy — `xtb_orbital` 0.8448 and
`rdkit_physchem` 0.8513, both below baseline. Broadcasting the same values per token
reversed the sign in both cases (`orbital_tok` 0.8702, `physchem_tok` 0.8744). The
superseded runs are kept under `runs_7251/_molecule_branch/` because they document
this architectural finding.

## Testing

```bash
export PYTHONPATH=src
python -m downstream.toxicity.xai.selftest
```

30 checks covering the SMILES character walk, atom-value spreading across
multi-character element symbols, Spearman correlation, curve AUC, the exact
McNemar implementation, and — in the torch/RDKit tier — that the attribution
code's token-matrix construction reproduces `features.build_token_feature_matrix`
exactly.

Descriptor data integrity is checked separately by `scripts/verify_7251.py`,
which verifies for all 7,251 compounds that atom composition and atom *ordering*
in each xTB output match the tokenised SMILES.

## Citation

Manuscript in preparation. Please check back for a citation, or contact the authors.

## Licence

Code: MIT (see `LICENSE`).
GFN2-xTB output data: CC-BY-4.0, via the Zenodo archive.

## Acknowledgements

Helmholtz Munich, Technical University of Munich, MN-AM, and Lhasa Limited.
Supported by the BIGCHEM programme. MolEncoder pre-trained weights from
[fabikru/MolEncoder](https://github.com/fabikru/MolEncoder).
