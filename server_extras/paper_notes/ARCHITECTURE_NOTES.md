# Architectural changes and why they were made

This file records four defects found in the descriptor-fusion pathway, each of which
had produced a false negative — a descriptor set appearing not to help, or appearing
to hurt, for reasons that turned out to be architectural rather than chemical.

They are documented because the negative results they produced were, at various
points, believed to be findings about chemistry. They were not.

## 1. Gradient starvation in the fusion pathway

**Symptom.** Adding descriptors changed accuracy very little in either direction.

**Measurement.** Gradient norms in the descriptor pathway were 0.67–1.15% of those in
the base encoder pathway. The fusion parameters were effectively frozen at their
initialisation while the encoder adapted around them.

**Fix.** A separate, higher learning rate for fusion parameters
(`--fusion-learning-rate 2e-3`) and a neutral gate initialisation
(`--gate-bias-init 0.0`). Fusion parameters are identified by name in
`FUSION_PARAM_MARKERS` in `train.py`.

**Effect.** Descriptors began to influence predictions at all.

## 2. Per-molecule normalisation destroyed descriptor semantics

**Symptom.** Molecule-level descriptors behaved erratically.

**Measurement.** LayerNorm was applied across the descriptor vector *within each
molecule*, so a molecule's HOMO energy was normalised against its own LUMO energy and
electrophilicity index — quantities on unrelated scales. Correlation between raw and
normalised values: 0.215 for HOMO, and **−0.380** for chemical potential. The sign
had inverted; higher raw values became lower normalised ones.

**Fix.** Dataset-level standardisation (`MOLECULE_FEATURE_STATS` in `features.py`,
means and standard deviations computed once across all 7,251 compounds) and
`nn.Identity()` in place of `nn.LayerNorm` at five sites in `finetune.py`.

**Effect.** Prerequisite for anything else to work. Molecule-level descriptors became
interpretable to the model.

## 3. Molecule-level descriptors in a separate branch

**Symptom.** Orbital and physicochemical descriptors *reduced* accuracy below
baseline — `xtb_orbital` 0.8448 (−0.0124) and `rdkit_physchem` 0.8513 (−0.0059), with
0/5 folds improving in both cases. This looked like a clean per-atom vs global
dichotomy: local descriptors help, global descriptors hurt.

**Diagnosis.** It was not a property of global descriptors. It was the separate fusion
branch they passed through. A descriptor that carries real information should at worst
be ignored by the model; it should not be able to make predictions worse. That it did
was the signal that something was wrong with the attachment, not the chemistry.

**Fix.** Broadcast each molecule-level value to every token
(`_broadcast_molecule_value` in `features.py`), so global descriptors enter through the
same pathway as per-atom ones. Registered as token features `orbital_homo`,
`orbital_lumo`, `orbital_electrophilicity` and ten `physchem_*` features.

**Effect.** Sign reversal in both cases: orbital −0.0124 → +0.0129, physchem −0.0059
→ +0.0172. The per-atom/global dichotomy was an artefact and does not appear in the
final results.

The orbital descriptor set was also reduced from six values to three — HOMO, LUMO and
electrophilicity index — since the remaining three (gap, chemical potential, hardness)
are algebraic functions of the first two.

## 4. Token features share a single attention weight

**Symptom.** Combining descriptors produced no gain over a single descriptor.

**Measurement.** `gasteiger` 0.874647 vs `gast_neigh_mr` 0.874626 — a difference of
0.00002 across three descriptors versus one. The additional descriptors were having no
effect whatsoever.

**Diagnosis.** In `finetune.py`, attention over tokens is computed from
`feature_strength = token_features.abs().sum(dim=-1)`, which sums *across* the feature
dimension. Every descriptor at a given token therefore receives the same attention
weight; the model cannot attend to charge at one atom and refractivity at another.

**Fix.** `run_perfeat.py`, a standalone `PerFeatureAttentionClassifier` computing one
attention distribution per feature and gathering a separate context vector for each.
Deliberately implemented outside `src/` so that no working code was modified.

**Effect.** Single-descriptor ceiling rose from 0.8670 to 0.8760. Combinations still
produced no gain, which is now interpretable as genuine redundancy rather than an
architectural artefact.

## Data cleaning

Fallback descriptor values (used when an xTB calculation was unavailable for a
compound) were removed entirely. The dataset was restricted to compounds having a
real GFN2-xTB output file, dropping four of 7,255.

Effect on the semi-empirical vs empirical comparison: with fallbacks present,
Gasteiger appeared to beat xTB by +0.0075 (p = 0.014). On clean data the gap closed to
+0.0007 (p = 0.41). The apparent advantage had been an artefact of contamination —
some "xTB" runs were partly using Gasteiger values.

## A note on the `pf_` runs

`run_perfeat.py` replicates the parent classifier's `__init__` layer-for-layer rather
than calling it, because constructing layers in a different order displaces the RNG
state and changes initialisation. An earlier version rebuilt `token_proj` after the
parent had already built it, yielding 0.8711 instead of 0.8746 for an otherwise
identical configuration. Verified after the fix: zero differing parameters at F = 1,
and `pf_gasteiger` reproduces the standard architecture's result.
