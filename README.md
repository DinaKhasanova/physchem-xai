# Physicochemical descriptors for explainable mutagenicity prediction

Code, data and results for

> Khasanova, D.; Tetko, I. V.; Menezes, F.; Ballabio, F.; Schwab, C. H.; Sacher, O.;
> Magdziarz, T.; Barber, C. *From black boxes to chemical interpretability:
> explainable deep learning for mutagenicity prediction.* (submitted)

A pretrained SMILES transformer (MolEncoder) is fine-tuned for Ames mutagenicity
prediction. Physicochemical descriptors (Gasteiger or GFN2-xTB charges, atomic
polarizabilities, heavy-atom neighbourhood counts, frontier-orbital descriptors and
RDKit molecular properties) are added through a confidence-gated correction to a
retained SMILES-only baseline. The question is whether descriptors that carry
explicit chemistry improve both accuracy and the chemical plausibility of
integrated-gradients attributions.

## Repository layout

```
data/
  ames_7251.csv                            Ames dataset (TDC), 7,251 compounds used in the study
  clean_data.py                            RDKit cleaning pipeline (salts, stereo, tautomers, duplicates)
  pesticides_genotoxicity_ames_format.csv  Pesticides genotoxicity dataset (Table 2)
  pesticides_dropped_organometallic.csv    organometallic compounds removed from it
  pesticides_external.csv                  11,591 pesticide compounds absent from Ames (Table 3)
  screening_results-ames.xls               OCHEM ToxAlerts screening export, Ames set
  screening_results-pesticides.xls         OCHEM ToxAlerts screening export, pesticide set
  xtb_7251/                                GFN2-xTB output files, see data/xtb_7251/README.md
AMES_xTB/                 processed xTB descriptor tables (also on Zenodo)
toxicophores/             29 Kazius structural alerts (SMARTS)
src/molencoder/           MolEncoder model definition
src/downstream/toxicity/  fine-tuning, descriptor construction, cross-validation, XAI tools
scripts/                  reproduction and analysis scripts
scripts/xai/              integrated gradients, flip cohorts, gate decomposition, figures
run_perfeat.py            per-feature attention variant (runs prefixed pf_)
run_ig_*.sh               integrated-gradients pipelines
run_external.sh           external validation on the pesticide compounds
runs_7251/                cross-validation results on Ames, one folder per descriptor set
runs_pest/                cross-validation results on the pesticide set
results/xai/              attribution and gate results, Ames
results/xai_pest/         attribution results and figure panels, pesticides
results/external/         external validation predictions and summary (Table 3)
```

Every run folder contains `cross_validation_summary.{json,csv}`,
`cross_validation_fold_metrics.csv` and, per fold, `test_metrics.{json,csv}` and the
exact train/validation/test split (`splits/`). Model checkpoints are too large for the
repository and are distributed as a GitHub release (see below).

## Paper tables and the folders behind them

All models: stratified 5-fold cross-validation, seed 42, identical folds across runs.

**Table 1 – Ames (7,251 compounds)**

| Paper row | Folder | ROC AUC |
|---|---|---|
| OCHEM-derived toxicophore assignment | `src/downstream/toxicity/evaluate_ochem_cv.py` | 0.789 |
| MolEncoder baseline, SMILES only | `runs_7251/baseline` | 0.857 |
| + polarizability (Crippen/RDKit) | `runs_7251/rdkit_mr` | 0.865 |
| + polarizability (xTB) | `runs_7251/xtb_polar` | 0.866 |
| + charges (Gasteiger) + neighborhoods + polarizability (Crippen) | `runs_7251/pf_gast_neigh_mr` | 0.870 |
| + orbital descriptors (xTB) | `runs_7251/orbital_tok` | 0.870 |
| + neighborhoods | `runs_7251/neighbors` | 0.870 |
| + charges (Mulliken/xTB) | `runs_7251/xtb_charge` | 0.874 |
| + global descriptors (RDKit) | `runs_7251/physchem_tok` | 0.874 |
| + charges (Gasteiger) | `runs_7251/pf_gasteiger` | 0.875 |
| + charges (Gasteiger) + neighborhoods | `runs_7251/pf_gast_neigh` | 0.876 |

Paired t-tests against the baseline are in `runs_7251/table1_statistics.csv`
(`scripts/compare_runs_stats.py`).

**Table 2 – Pesticides genotoxicity:** `runs_pest/{baseline, gasteiger, neighbors,
gast_neigh, physchem_tok}`.

**Table 3 – External validation:** `results/external/external_summary.csv`, produced by
`run_external.sh` from the five Ames-trained fold models of each descriptor set.

**Figures 4 and 5, Tables S1 and S2:** `results/xai/` and `results/xai_pest/`, produced
by `run_ig_gast.sh`, `run_ig_xtb.sh`, `run_ig_pest.sh`,
`scripts/xai/gate_decomposition.py` and `scripts/xai/gate_by_label.py`.

### Per-feature attention (`pf_` runs)

In the standard classifier, token hidden states are pooled with one weight per token,
computed from the summed magnitude of all its descriptors. With several descriptors
the model therefore cannot weight them separately. `run_perfeat.py` gives each
descriptor its own attention weight and pooled context vector; nothing else in the
training pipeline changes. With a single descriptor the two variants are
mathematically identical: `pf_gasteiger` and `gasteiger` give the same ROC AUC
(0.875 ± 0.011) and balanced accuracy within 0.002 (0.791 vs 0.789), the difference
coming from non-deterministic CPU training.
All other runs use the standard pooling. Integrated gradients were computed on the
`gasteiger` checkpoints, which are also the ones provided for the Gasteiger model.

### Descriptor attachment and scaling

Per-atom values are written onto the SMILES token(s) of that atom; molecule-level
values are broadcast to every token, so both enter through the same pathway.
Molecule-level descriptors are standardised with dataset-level means and standard
deviations (`MOLECULE_FEATURE_STATS` in `features.py`). The descriptor pathway is
trained with its own learning rate (2 × 10⁻³) and a neutral gate initialisation.

`runs_7251/_molecule_branch/` holds an earlier design in which molecule-level
descriptors entered through a separate branch. It is not used in the paper and is kept
for reference only (metrics, no checkpoints).

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .              # or: pip install -r requirements-lock.txt
export PYTHONPATH=src
```

MolEncoder pretrained weights are downloaded from Hugging Face on first use
([fabikru/MolEncoder](https://github.com/fabikru/MolEncoder)).
`scripts/setup_vm.sh` provisions a fresh Linux machine. Training is CPU-feasible; the
full Table 1 took about a day on 2 vCPUs.

## Reproducing the results

```bash
python scripts/verify_7251.py          # checks xTB outputs against the tokenised SMILES
bash scripts/run_table_7251.sh         # Table 1, standard pooling
python run_perfeat.py ...               # Table 1, pf_ runs; exact commands in the file header
bash scripts/run_pesticides.sh         # Table 2
bash run_external.sh                   # Table 3
bash run_ig_gast.sh                    # integrated gradients, Ames, Gasteiger
bash run_ig_xtb.sh                     # integrated gradients, Ames, xTB charges
bash run_ig_pest.sh                    # integrated gradients, pesticides
```

To reuse the trained models instead of retraining, download the checkpoints:

```bash
bash scripts/download_checkpoints.sh   # needs the GitHub CLI (gh); verified against ckpt_md5.txt
```

Checkpoints are provided for every model in Tables 1–3 except the two polarizability
sets (`rdkit_mr`, `xtb_polar`), which did not improve significantly over the baseline
and were not retained. Their per-fold metrics and splits are in the repository, and
they can be retrained with `scripts/run_table_7251.sh`.

## Tests

```bash
python -m downstream.toxicity.xai.selftest --pure   # no torch/RDKit needed
python -m downstream.toxicity.xai.selftest          # adds the environment check
```

Checks the SMILES character walk, the assignment of atom values to multi-character
element symbols, the Spearman and curve-AUC helpers, and checkpoint reading.

## Data availability

- **Ames:** Therapeutics Data Commons (TDC). The raw set was cleaned with
  `data/clean_data.py`. Of the 7,255 compounds sent to 3D structure generation, four did
  not yield a valid protonated structure with Epik and were excluded, leaving 7,251; they
  are listed in `data/xtb_7251/ames_out_assignment.csv`.
- **GFN2-xTB descriptors (AMES_xTB):** Zenodo,
  [10.5281/zenodo.22919496](https://doi.org/10.5281/zenodo.22919496), with a copy in
  `AMES_xTB/` and the raw output files in `data/xtb_7251/`.
- **OCHEM ToxAlerts screening:** exported tables in `data/screening_results-*.xls`.

## Licence

Code: MIT (see `LICENSE`). GFN2-xTB descriptor data: CC-BY-4.0 (Zenodo).

## Acknowledgements

Helmholtz Munich, Technical University of Munich, MN-AM and Lhasa Limited. Supported by
the BIGCHEM programme. MolEncoder pretrained weights from
[fabikru/MolEncoder](https://github.com/fabikru/MolEncoder).
