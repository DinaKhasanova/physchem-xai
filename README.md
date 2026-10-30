# From Black Boxes to Chemical Interpretability: Explainable Deep Learning for Mutagenicity Prediction

Code, data and results for a study of whether physicochemical descriptors make
transformer-based mutagenicity predictions more chemically interpretable, rather than
merely more accurate. A pretrained SMILES language model ([MolEncoder](https://huggingface.co/fabikru/MolEncoder))
is augmented with descriptors through a confidence-gated correction to a retained
SMILES-only baseline, and evaluated on two Ames-endpoint datasets of 7,251 and 18,754
compounds.

## Quick Start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
export PYTHONPATH=src
```

MolEncoder weights are downloaded from Hugging Face on first use. To reuse the trained
models instead of retraining, `bash scripts/download_checkpoints.sh` fetches them from
the GitHub release and verifies them against `ckpt_md5.txt`. Training is CPU-feasible;
the full Ames table took about a day on 2 vCPUs. `RESULTS.md` maps every table and
figure in the paper onto the folder that produced it.

## Repository Structure

### Core Implementation (`src/`)

- **`molencoder/`** — the pretrained encoder.
- **`downstream/toxicity/`** — fine-tuning, descriptor construction and cross-validation.
  `features.py` builds and scales the descriptor channels, `train.py` holds the gated
  fusion, and `run_cross_validation.py` is the entry point for every run in the paper.

### Analysis (`scripts/`)

- **`run_table_7251.sh`**, **`run_pesticides.sh`**, **`run_external.sh`** (repo root) — the three
  result tables, in that order. **Start here!**
- **`xai/integrated_gradients.py`** — per-atom attributions, 64-step midpoint rule, with
  a completeness check on every molecule.
- **`xai/attribution_shift.py`** — how far each charge scheme moves attribution onto
  alert-matched atoms, with the paired test between the two.
- **`xai/gate_decomposition.py`**, **`xai/gate_by_label.py`** — the descriptor pathway's
  own contribution, read off the model and cut by true label and alert.
- **`compare_runs_stats.py`** — paired t-tests against the SMILES-only baseline.
- **`verify_7251.py`** — checks the xTB outputs against the tokenised SMILES.

### Results (`runs_*/`, `results/`)

- **`runs_7251/`**, **`runs_pest/`** — cross-validation output, one folder per descriptor
  set. Each holds `cross_validation_summary.{json,csv}` and, per fold, `test_metrics.{json,csv}`
  and the exact train/validation/test split.
- **`results/external/`** — predictions and summary for the 11,591 pesticide compounds
  absent from the Ames set.
- **`results/xai/`**, **`results/xai_pest/`** — attributions, flip cohorts and figure panels.

Two sets of runs are kept for reference but are not used in the paper: `pf_*` gives each
descriptor its own attention weight instead of one weight per token (identical to the
standard pooling when there is only one descriptor), and `_molecule_branch/` is an
earlier design in which molecule-level descriptors entered through a separate branch.

### Data (`data/`)

- **`ames_7251.csv`** — Ames set from Therapeutics Data Commons, cleaned with
  `clean_data.py`. Four of 7,255 compounds yielded no valid protonated structure and
  were excluded.
- **`pesticides_genotoxicity_ames_format.csv`** — pesticide genotoxicity set;
  **`pesticides_external.csv`** is the subset absent from Ames.
- **`xtb_7251/`** — raw GFN2-xTB outputs. Processed descriptor tables are in
  **`AMES_xTB/`** and on Zenodo.
- **`screening_results-*.xls`** — OCHEM ToxAlerts screening exports.
- **`toxicophores/alerts_kazius29.csv`** (repo root) — the 29 Kazius structural alerts as SMARTS.

## Citation

```bibtex
@unpublished{khasanova2026interpretability,
  title  = {From Black Boxes to Chemical Interpretability: Explainable Deep
            Learning for Mutagenicity Prediction},
  author = {Khasanova, D. and Menezes, F. and Ballabio, F. and Schwab, C. H.
            and Sacher, O. and Magdziarz, T. and Barber, C. and Tetko, I. V.},
  year   = {2026},
  note   = {Submitted}
}
```

## Links

- GFN2-xTB descriptors (AMES_xTB): [10.5281/zenodo.22919496](https://doi.org/10.5281/zenodo.22919496)
- MolEncoder weights: [fabikru/MolEncoder](https://huggingface.co/fabikru/MolEncoder)
- MolEncoder code and paper: [FabianKruger/molencoder](https://github.com/FabianKruger/molencoder), [10.1039/D5DD00369E](https://doi.org/10.1039/D5DD00369E)

## License

Code: MIT (see `LICENSE`). GFN2-xTB descriptor data: CC-BY-4.0 (Zenodo).

## Acknowledgements

Helmholtz Munich, Technical University of Munich, MN-AM and Lhasa Limited, supported by
the BIGCHEM programme.
