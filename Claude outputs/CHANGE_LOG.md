# Change log — Article2-XAI_DFT_2_revised.docx

Every change is in **dark blue** in the document. Black text is yours, untouched.
Verified run by run against your original upload: 0 changed runs left unmarked, 0 original runs wrongly marked.
Each entry names the file in `physchem-xai` that justifies it.

---

## 1. XAI methods rewritten for integrated gradients

**Where:** *XAI methods* section (four paragraphs + the equation image).
**Evidence:** `scripts/xai/integrated_gradients.py`, `run_ig_gast.sh`.

The section described occlusion end to end. Replaced with integrated gradients as actually implemented:

| Detail in new text | Source in code |
|---|---|
| Baseline for token embeddings = pad-token embedding | `base_embed = embed(torch.full_like(input_ids, pad_id))` |
| Baseline for descriptors = per-feature median of that fold's train split | `descriptor_baseline … .median(dim=0)` |
| 64 steps, midpoint rule | `--steps 64`; `alpha = (step + 0.5) / steps` |
| Token and descriptor pathways attributed separately | returns `attr_embed`, `attr_tf` |
| Completeness used as convergence check | `completeness = |attr.sum() − (f_x − f_0)|` |
| Per-atom value = token + descriptor pathway | `"attr_total": float(a + b_)` |

The occlusion equation image (`image4.png`) was replaced with a rendered integrated-gradients equation.

**Also removed:** the global-descriptor occlusion paragraph. No IG run used molecule-level descriptors — `results/xai/` and `results/xai_pest/` contain only baseline and Gasteiger attributions. Replaced with a statement of what the attribution analysis actually covers.

---

## 2. Three 3D-structure workflows reduced to CORINA Classic

**Where:** *Molecular Preparation* (Methods) and the workflow paragraph in Results.

- Results paragraph: the comparison of three workflows (CORINA / custom RDKit / LigPrep + MacroModel) is gone. Citations 29, 37, 38 and 33 are preserved and re-attached to the single remaining workflow.
- Methods: *"Protonation states were generated from SMILES using LigPrep"* → *"assigned to the CORINA-generated three-dimensional structures using LigPrep"*, so LigPrep is no longer implied to build the 3D structures. This now matches Figure 2.

No RDKit 3D-embedding or MacroModel code exists anywhere in the repository (`grep` for `EmbedMolecule`, `ETKDG`, `MacroModel` returns nothing).

---

## 3. Stale numbers corrected

| Location | Was | Now | Source |
|---|---|---|---|
| Results, baseline | 0.864 ± 0.007 / 0.786 ± 0.007 | **0.857 ± 0.01 / 0.776 ± 0.01** | `runs_7251/baseline/cross_validation_summary.json` → 0.8572 / 0.7755 |
| Results, OCHEM | 0.790 ± 0.005 / 0.702 ± 0.004 | **0.789 ± 0.01 / 0.702 ± 0.01** | matched to Table 1 — see open question 2 |
| Results | "OCHEM ToxAlters tool" | **ToxAlerts** | typo |

**All 15 rows of Tables 1 and 2 were verified against the run summaries: zero mismatches.** Tables were not modified.

---

## 4. Figure references fixed

- "Figs. 2 and 3" → **Figs. 3 and 4**
- "Figure N" → **Figure 3**
- "(Fig. 1)" deleted from the ToxAlerts paragraph — no such figure exists; per your instruction the paragraph now makes the general point (same fragment, different environment) without referring to two specific molecules or a figure. Your sentences are otherwise preserved verbatim.

All figure and table references now resolve: Figures 1–4, Tables 1, 2, S1.

---

## 5. Figure 3 lead-in paragraph rewritten

**Was:** two molecules, occlusion, "a nitroaniline derivative was correctly classified as toxic by both models."

Three errors: Figure 3 shows **three** molecules; the compound is 2-nitro**anisole**, not a nitroaniline; and the baseline got it **wrong**.

**Evidence:** `results/xai/flip_cohort.csv` — all three figure molecules are `stratum=flip_to_correct`, `label=1`, `pred_baseline=0`, `pred_descriptor=1`. The caption you kept ("In each case the SMILES-only model misclassifies the compound") is correct; the paragraph contradicted it.

Rewritten for three molecules, integrated gradients, and the three mechanisms in the caption.

---

## 6. Descriptor standardisation described accurately

**Where:** end of *Atom-to-Token Feature Mapping*.
**Evidence:** `run_table_7251.sh`, `features.py:568–588`.

The text said all channels were dataset-standardised. In fact single-descriptor runs use `--token-feature-normalize-mode none` and only combined runs use `dataset`. Now stated correctly, with the reason from the code comment: without scaling, molar refractivity (mean 3.24) and neighbourhood counts (1.7–2.3) swamp partial charges (mean |q| 0.041) by 50–80× in the pooling weight. Added the charge clipping (−2 to 2) and the log transform on neighbourhood counts.

The 0.215 / −0.380 figures are correct (`features.py:601`) — I added "measured on 400 random molecules", which the same comment states.

---

## 7. The xTB-fallback sentence (what the draft called ORCA)

Rewritten as you described, without numbers. Confirmed by `data/xtb_7251/README.md`:

> *"an earlier version of the dataset substituted Gasteiger charges where xTB output was missing, which contaminated the comparison between semi-empirical and empirical descriptors."*

The new sentence makes your point — consistency of the charges matters more than their accuracy — and adds that no substitute values are used in the present work, which the README states explicitly.

---

## 8. Discussion moved after Results, opening paragraph replaced

The opening claimed **early fusion**, a **zero-initialised descriptor MLP**, and that **all molecules start from a SMILES-only baseline**. All three contradict the code:

- `SmilesAnchorResidualAuxClassifier.forward` calls `self.encoder(input_ids=…)`. The transformer never receives descriptors.
- The only zero-initialised tensor is `gate_mlp[-1].weight` (`finetune.py:342`). `token_proj`, `aux_proj`, `aux_classifier` use default init.
- With `--gate-bias-init 0.0` the gate is a constant 0.5 at initialisation, so training starts with a half-strength descriptor correction, not a SMILES-only baseline. Your **Methods section says this correctly** and the Discussion contradicted it.

Replaced with the architecture as coded, and justified with evidence from the repo:
- early fusion was tried and abandoned — injection weights reached ~0.1 % of token-embedding magnitude (`train.py:200–215`);
- the molecule-level fusion branch dropped ROC AUC to 0.845 and 0.851 against a 0.857 baseline, reversed to 0.870 and 0.874 by per-token broadcasting (`RESULTS.md`, `runs_7251/_molecule_branch/`).

Your CLS-pooling limitation paragraph is unchanged (it already referred to integrated gradients correctly).

**Added** a limitations paragraph: the gate is conditioned only on baseline confidence and margin, and across all 7,251 compounds has mean 0.25, sd 0.06, with the mean differing by **less than 0.003** between compounds the descriptors correct and compounds they damage (computed from `results/xai/gate_decomposition.csv`).

> Note: `RESULTS.md` says the gate "settles at 47–48 %". The stored decomposition gives 0.25. I used the measured value.

---

## 9. New Results paragraph: alert-level analysis

**Evidence:** `results/xai/gate_by_label.csv`, `scripts/xai/gate_by_label.py`.

Generalises your three anecdotes to the whole dataset. Because the model is additive at the output, the descriptor pathway's contribution is exactly `gate × aux_delta` — verified: max deviation 7 × 10⁻⁷ across all 7,251 rows.

Across 4,026 alert-bearing compounds: **+1.53** mean contribution for mutagenic vs **0.00** for non-mutagenic. Per-alert numbers quoted for epoxide, aromatic nitro and aromatic amine. The grouping uses only the experimental label and substructure membership, so it is not circular — a point the script's own docstring makes, and worth keeping in the text.

---

## 10. Conclusion added

Two paragraphs drafted from your own Results and Abstract wording. Every number verified. Please read it as a draft — it is the section most likely to need your voice.

---

## 11. Gradient-norm claim replaced with the step-size argument

**Was:** *"measured gradient norms in the fusion parameters were 0.67–1.15 % of those in the encoder pathway."*

Not reproducible. Measured with `scripts/xai/gradient_norms.py` (written for this, now in your repo), under a single global learning rate, over 80 training batches of fold 1:

| Gate initialisation | fusion / encoder gradient norm |
|---|---|
| `--gate-bias-init 0.0` (shipped) | min 246 %, median 525 %, max 1127 % |
| `--gate-bias-init -2.0` (suppressed) | min 66 %, median 152 %, max 257 % |

The claim is wrong by roughly two orders of magnitude, and in the opposite direction. `aux_classifier` is a randomly initialised linear layer wired straight to the loss, so it necessarily receives large gradients; the gate scales `aux_delta` at the output, which scales gradients into the auxiliary head but never lifts them below the encoder's.

**Replaced with the argument from `train.py:200–215`,** which is correct regardless of gradient magnitude: AdamW normalises each update by the running gradient magnitude, so a parameter's displacement is bounded by learning rate × number of steps. At 2 × 10⁻⁵ over ~1,270 steps that bound is ≈ 0.025.

The 1,270 figure checks out independently: `train.csv` holds 5,075 compounds, batch size 16 → 318 steps/epoch, and the best validation loss is reached at epoch 4 in 4 of 5 folds (`cross_validation_fold_metrics.csv`). 318 × 4 = 1,272.

The sentence now also states that gradient magnitude is *not* the limitation, which is what your two runs show.

---

## 12. Protonation counts resolved (reading B)

**Where:** *TDC Ames* (Methods) and *Molecular Preparation*.

Confirmed from the data on `dft`: `xtb_7251/outs/` holds 7,251 files, `smiles_to_out_stem.csv` 7,252 lines and `ames_out_assignment.csv` 7,256 lines — 7,251 indexed calculations for 7,255 compounds, with exactly 4 `NOT_CALCULATED`. `protonation_federico_corina_7246_xyz` does not exist on that machine. So no compound fell back to RDKit charges, and 7,246 cannot have been the xTB input.

Per your decision:

- *"A total of 7,255 molecules were processed with CORINA Classic (MN-AM)²⁹ … Subsequent protonation state assignment with Epik at pH 7.4 ± 0.2 yielded **7,251** valid protonated structures, all of which were carried through the GFN2-xTB calculations described below."* The 7,246 comparison sentence is deleted; citation 29 is preserved and now sits on the single workflow.
- *TDC Ames*: "Due to XTB convergence failures for 4 compounds" → **"Protonation state assignment failed for 4 compounds"**, so the two sections no longer disagree about where the four were lost.

### ⚠ This contradicts your dataset README — please fix it before the Zenodo release

`data/xtb_7251/README.md` says the opposite:

> *"Four of the 7,255 TDC compounds have no calculation, because GFN2-xTB did not produce a usable result — hypervalent species and non-convergent cases"*

and names them: Drug 6234 (`N#[N+]c1nc[nH]c1C(N)=O`), Drug 6447 (`OCP(Cl)(CO)(CO)CO`), Drug 7555 (`[O-][Cl+2]([O-])O`), Drug 864 (`[O-][Cl+3]([O-])([O-])[O-]`).

Under reading B these four failed **Epik**, not xTB, and xTB failed on nothing. That README is licensed CC-BY-4.0 and asks to be cited alongside the manuscript, so a reviewer comparing the two will see the disagreement. Either correct the README, or tell me B was wrong and I will switch the paper to reading A.

---

## 13. Supporting Information document

`Supporting_Information_TableS1.docx` — Table S1, all nine alerts, verified row by row against `gate_by_label.csv` (zero mismatches). **You will need to merge this into your existing SI**, which already contains the SMARTS of functional groups.

---

# Open questions — I could not resolve these

### 1. Unresolved note in the text
*"(add references to respective articles)"* is still in the OCHEM paragraph. I did not invent citations. Please add them or tell me to cut the parenthesis.

### 2. OCHEM results are not reproducible from the repo
`evaluate_ochem_cv.py` reads an `.xlsx` that is not in the repository, and no OCHEM output is stored. I harmonised the text to Table 1, but I could not verify either. The paper also states this file "is available in the project GitHub repository" — it is not. Either add it or amend the sentence.

### 3. Things I changed that you may want to check
- **Michael acceptor, Figure 3.** "does not identify any part of the conjugated system" was not quite right — the ketone oxygen is +0.045 in the baseline. Softened to "no meaningful positive attribution … the largest being +0.05 on the ketone oxygen".
- **Epoxide, Figure 4.** "attribution shifts away from the inert chain and toward the strained ring" did not match the data: chain atoms move positive (−0.91 → +0.02) and ring atom 4 moves to **−1.49**. Rewritten to say positive attribution concentrates on the site of attack while the more substituted ring carbon becomes the most negative atom — which is the better result, since it is what SN2 selectivity predicts.
- I removed the yellow highlighting from paragraphs I rewrote completely. Highlighting on untouched paragraphs is preserved.
- The method-selection sentence in XAI methods notes that IG loses no atoms. If you would rather not mention occlusion at all, cut that sentence — `integrated_gradients.py` records that occlusion's `--token-mode recompute` failed on 47.7 % of atoms, if you want the number instead.

---

# Verified correct — not changed

- All 15 rows of Tables 1 and 2, against `cross_validation_summary.json`.
- **All 25 per-atom attribution values** quoted for Figures 3 and 4, against `results/xai/ig_*.csv` and `results/xai_pest/ig_*.csv`.
- Dataset counts: 7,251 = 3,950 + 3,301; 18,754 = 7,638 + 11,116.
- p = 0.017 and "all 5 folds" for Gasteiger; the polarizability results being non-significant; orbital descriptors having the lowest fold-to-fold variance (sd 0.0074).
- Pesticide "p = 0.17" for Gasteiger — recomputed as 0.1698. My method reproduces your stored `table1_statistics.csv` p-values to five decimals, so the pesticide statistics are sound. Neighbourhoods p = 0.034 and RDKit global p = 0.044 are both significant, as the text says.
- The fusion learning rate 2 × 10⁻³ and gate bias initialisation (g = 0.5 vs 0.12).
- The 0.215 / −0.380 per-molecule normalisation figures.
