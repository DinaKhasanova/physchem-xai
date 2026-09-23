# GFN2-xTB calculations for the Ames dataset

7,251 single-point GFN2-xTB calculations, one per compound in the filtered Ames
mutagenicity dataset.

## Contents

```
outs/                        7,251 output files, mol1.out … molN.out
smiles_to_out_stem.csv       canonical SMILES → output file stem (machine-readable index)
ames_out_assignment.csv      full provenance: Drug_ID, SMILES, label, output file, match method
```

`smiles_to_out_stem.csv` is read at training time by `_build_xyz_index` in
`src/downstream/toxicity/features.py`. Its columns are `canonical_smiles,out_stem`;
the stem resolves to `outs/<stem>.out`.

`ames_out_assignment.csv` is the human-readable record of how each Ames compound was
matched to its calculation. Nothing reads it during training. It lists all 7,255 TDC
compounds, four of which are marked as excluded.

## Provenance

Calculations were performed with GFN2-xTB as implemented in ULYSSES. Three-dimensional
structures were generated from SMILES with CORINA Classic (MN-AM), protonation states
were assigned with Schrödinger Epik at pH 7.4 ± 0.2, and the resulting geometries were
optimised prior to the single-point calculation.

Three descriptor families are parsed from these files:

| Descriptor | Level | Used in run |
|---|---|---|
| Mulliken partial charges | per atom | `xtb_charge`, `neighbors` |
| Atomic polarizabilities | per atom | `xtb_polar` |
| HOMO, LUMO, electrophilicity index | per molecule | `orbital_tok` |

## Excluded compounds

Four of the 7,255 TDC compounds have no calculation. Protonation state assignment did
not return a usable structure for them, so they never entered the GFN2-xTB step —
hypervalent species and other unusual oxidation states:

| Drug_ID | SMILES |
|---|---|
| Drug 6234 | `N#[N+]c1nc[nH]c1C(N)=O` |
| Drug 6447 | `OCP(Cl)(CO)(CO)CO` |
| Drug 7555 | `[O-][Cl+2]([O-])O` |
| Drug 864 | `[O-][Cl+3]([O-])([O-])[O-]` |

No fallback or substitute values are used anywhere. Every compound in the training set
has a real calculation. This was a deliberate choice: an earlier version of the dataset
substituted Gasteiger charges where xTB output was missing, which contaminated the
comparison between semi-empirical and empirical descriptors.

## Integrity

The index and the provenance file agree on every compound-to-calculation link, every
calculation has a provenance record, and no two compounds share a calculation.
Verified with:

```bash
python scripts/verify_7251.py
```

which additionally checks, for all 7,251 compounds, that the atom composition and the
atom *ordering* in each output file match the SMILES string as tokenised.

## Licence

CC-BY-4.0. If you use these calculations, please cite the Zenodo archive and the
accompanying manuscript.
