# Results

All runs: 5-fold stratified cross-validation, 7,251 compounds, seed 42, MolEncoder
encoder (15.4M parameters, hidden size 384) with confidence-gated residual fusion
(`SmilesAnchorResidualAuxClassifier`). Numbers below are read directly from
`runs_7251/<run>/cross_validation_summary.json`.

## Table 1 — ROC-AUC by descriptor set

Sorted by mean ROC-AUC. Paired t-tests across the five folds against `baseline`;
t(4), two-tailed critical value 2.776 at α = 0.05.

| Run | Descriptors | ROC-AUC | Δ | t | folds improved |
|---|---|---|---|---|---|
| `baseline` | none (SMILES only) | 0.8572 ± 0.0102 | — | — | — |
| `rdkit_mr` | molar refractivity | 0.8649 ± 0.0175 | +0.0077 | 1.95 | 4/5 |
| `xtb_polar` | xTB polarizabilities | 0.8663 ± 0.0180 | +0.0091 | 1.86 | 4/5 |
| `pf_gast_neigh_mr` | Gasteiger + neighbours + MR (per-feature) | 0.8696 ± 0.0116 | +0.0123 | 3.60 | 5/5 |
| `gast_neigh` | Gasteiger + neighbours | 0.8701 ± 0.0120 | +0.0128 | 2.76 | 4/5 |
| `orbital_tok` | HOMO, LUMO, electrophilicity (broadcast) | 0.8702 ± 0.0074 | +0.0129 | 7.86 | 5/5 |
| `neighbors` | neighbour-shell charge aggregates | 0.8704 ± 0.0117 | +0.0132 | 4.05 | 5/5 |
| `xtb_charge` | GFN2-xTB Mulliken charges | 0.8739 ± 0.0098 | +0.0167 | 3.95 | 5/5 |
| `physchem_tok` | 10 RDKit properties (broadcast) | 0.8744 ± 0.0160 | +0.0172 | 5.17 | 5/5 |
| `gast_neigh_mr` | Gasteiger + neighbours + MR | 0.8746 ± 0.0120 | +0.0174 | 4.68 | 5/5 |
| `gasteiger` | Gasteiger partial charges | 0.8746 ± 0.0110 | +0.0174 | 3.94 | 5/5 |
| `pf_gasteiger` | Gasteiger (per-feature attention) | 0.8751 ± 0.0110 | +0.0178 | 4.03 | 5/5 |
| `pf_gast_neigh` | Gasteiger + neighbours (per-feature) | 0.8760 ± 0.0092 | +0.0187 | 9.56 | 5/5 |

Standard deviations are across folds. `roc_auc_se` and 95% confidence intervals are
in each run's JSON summary, along with balanced accuracy.

## Superseded: molecule-level fusion branch

`runs_7251/_molecule_branch/` — metrics only, no checkpoints.

| Run | ROC-AUC | Δ vs baseline | t | folds improved |
|---|---|---|---|---|
| `xtb_orbital` | 0.8448 ± 0.0140 | −0.0124 | −3.93 | 0/5 |
| `rdkit_physchem` | 0.8513 ± 0.0127 | −0.0059 | −2.92 | 0/5 |

These two runs attached molecule-level descriptors through a separate fusion branch
with per-molecule LayerNorm. Both *degraded* accuracy. Replacing the branch with
per-token broadcasting and dataset-level standardisation reversed the sign for the
same descriptors: `orbital_tok` +0.0129 and `physchem_tok` +0.0172. They are retained
as evidence for the architectural claim, not as results.

## Key comparisons

**Semi-empirical vs empirical charges.** `gasteiger` 0.8746 vs `xtb_charge` 0.8739 —
Δ = +0.0007, t = 0.91, 2/5 folds. Statistically indistinguishable. GFN2-xTB charges
confer no measurable advantage over Gasteiger charges for this task, despite the
several orders of magnitude difference in computational cost.

**Descriptor combinations are redundant.** `pf_gast_neigh` 0.8760 vs `pf_gasteiger`
0.8751 — Δ = +0.0009, t = 0.30, 3/5 folds. Adding neighbour-shell aggregates on top
of Gasteiger charges does not help. The information appears to be recoverable by the
encoder from the charges plus SMILES topology.

**Per-feature attention.** In the standard architecture, token features are collapsed
into a single attention weight via `feature_strength = token_features.abs().sum(-1)`,
so a model given three descriptors cannot weight them differently. Measured effect:
`gasteiger` 0.874647 vs `gast_neigh_mr` 0.874626 — a difference of 0.00002, i.e. the
additional descriptors were being ignored. `run_perfeat.py` implements per-feature
attention weights as a standalone variant. It raises the ceiling for single-descriptor
runs (0.8670 → 0.8760) but does not produce a gain from combining descriptors, which
is what establishes the redundancy result above.

## Known limitations

- The confidence gate is blind to descriptor content: it settles at 47–48% for
  descriptor sets that help by +0.017 and for ones that hurt by −0.017 alike.
- Auxiliary representations are averaged (`aux_parts` mean) before fusion, which may
  dilute descriptor sets of differing informativeness.
- The per-feature variant retains a `token_proj` bottleneck.
- Two-character element symbols (Cl, Br) have their atom value split evenly across
  both SMILES characters (`value / 2.0` each). This preserves the per-molecule sum
  but means neither character carries the full atomic value.
