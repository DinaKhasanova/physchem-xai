# AMES_xTB: GFN2-xTB quantum-chemical descriptors for the Ames mutagenicity dataset

AMES_xTB provides semi-empirical quantum-chemical descriptors (GFN2-xTB) for
7,251 compounds from the Ames mutagenicity benchmark of the Therapeutics Data
Commons (TDC). The data are split into two linked tables:

- a **molecule-level table** with the SMILES, the Ames label and global
  descriptors (frontier-orbital energies, electrophilicity index, total energy,
  total polarizability), and
- a **token-level table** with the SMILES split into MolEncoder tokens, where
  each token carries the local (atom-level) descriptor values mapped onto it.

These are the descriptor values used to train the descriptor-augmented
MolEncoder models in the accompanying study. All values are reported as parsed
from the GFN2-xTB output files, with no standardisation or rescaling.

## Files

| File | Rows | Size | Content |
|---|---|---|---|
| `AMES_xTB_molecules.csv` | 7,251 | 1.2 MB | one row per compound |
| `AMES_xTB_tokens.csv` | 220,616 | 7.3 MB | one row per SMILES token |
| `README.md` | | | this file |

Both tables are UTF-8, comma-separated, and have a header row. They are linked
by `Drug_ID`, the TDC compound identifier. Rows are in the same compound order
in both files.

## Molecule-level table: `AMES_xTB_molecules.csv`

| Column | Type | Unit | Description |
|---|---|---|---|
| `Drug_ID` | string | | TDC compound identifier (primary key) |
| `smiles` | string | | SMILES as used in the study. This is the exact string that was tokenised; the token table is derived from it |
| `canonical_smiles` | string | | RDKit canonical SMILES used to link the compound to its xTB calculation |
| `ames_label` | int | | Ames test result: 1 = mutagenic (positive), 0 = non-mutagenic (negative) |
| `xtb_output_file` | string | | name of the GFN2-xTB output file for this compound |
| `xtb_total_charge` | int | e | total molecular charge of the xTB calculation (see *Protonation states*) |
| `n_heavy_atoms` | int | | number of non-hydrogen atoms |
| `n_atoms_total` | int | | number of atoms in the xTB calculation, including hydrogens of the calculated protonation state |
| `n_tokens` | int | | number of SMILES tokens (rows in the token table) |
| `homo_Eh` | float | Hartree | energy of the highest occupied molecular orbital ¹ |
| `lumo_Eh` | float | Hartree | energy of the lowest unoccupied molecular orbital ¹ |
| `homo_lumo_gap_Eh` | float | Hartree | LUMO − HOMO |
| `electrophilicity_index_Eh` | float | Hartree | ω = μ² / (2η), with μ = (E_HOMO + E_LUMO)/2 and η = E_LUMO − E_HOMO ¹ |
| `total_energy_Eh` | float | Hartree | GFN2-xTB total energy of the single-point calculation |
| `total_polarizability_bohr3` | float | bohr³ (a.u.) | molecular static dipole polarizability; equals the sum of the atomic polarizabilities of all atoms, including hydrogens |

¹ Used as model input in the study (run `orbital_tok`).

HOMO and LUMO are taken from the orbital energies after SCF convergence. The
HOMO is the highest orbital with non-zero occupation, and the LUMO is the next
orbital.

## Token-level table: `AMES_xTB_tokens.csv`

| Column | Type | Unit | Description |
|---|---|---|---|
| `Drug_ID` | string | | TDC compound identifier (links to the molecule table) |
| `token_index` | int | | 0-based position of the token in `smiles` |
| `token` | string | | the token (one SMILES character) |
| `atom_index` | int or empty | | 0-based index of the heavy atom the token belongs to; empty for bond, branch, ring-closure and stereo tokens |
| `element` | string or empty | | element symbol of that atom |
| `xtb_charge_e` | float | e | GFN2-xTB atomic partial charge mapped onto the token ² |
| `xtb_polarizability_bohr3` | float | bohr³ (a.u.) | GFN2-xTB atomic static dipole polarizability mapped onto the token ² |

² Used as model input in the study (runs `xtb_charge` and `xtb_polar`).

`atom_index` follows the atom order of the SMILES. This is also the RDKit atom
order and the order of the heavy atoms in the xTB output file.

### How descriptor values are mapped onto tokens

MolEncoder uses a character-level tokenizer, so each SMILES character is one
token. Joining the `token` column in `token_index` order gives back `smiles`
exactly. The mapping below is the one used to build the model input.

- **Single-letter atoms** (`C`, `c`, `N`, `n`, `O`, `S`, `F`, `I`, ...) carry
  the full value of their atom.
- **Two-letter halogens outside brackets** (`Cl`, `Br`) are two tokens. The
  atom's value is split equally between them, so each token carries half.
- **Bracket atoms** (for example `[nH]`, `[N+]`, `[O-]`): every character of
  the bracket has the same `atom_index`. The full value is on the first letter
  of the element symbol, and the other characters in the bracket carry 0.0.
- **Other tokens** (bonds `= # / \`, branches `( )`, ring-closure digits,
  stereo `@`) carry 0.0 and have an empty `atom_index`.

To get one value per atom, sum over the tokens of that atom:

```python
atom_charge = tok.groupby(["Drug_ID", "atom_index"])["xtb_charge_e"].sum()
```

For every atom in the dataset, this sum reproduces the value in the xTB output
exactly.

The model input also contains a `[CLS]` token at the start and a `[SEP]` token
at the end. Both carry zeros and are not listed, so the model position of a
token is `token_index + 1`.

Hydrogens are not SMILES atoms in this dataset. Their charges and
polarizabilities are therefore not in the token table, but they are included
in `total_polarizability_bohr3` and in the raw output files.

## Calculation details

- Method: GFN2-xTB, as implemented in ULYSSES [TODO: version]
- Geometries: generated from SMILES and optimised with GFN2-xTB before the
  single-point calculation [TODO: conformer generation tool and settings]
- Total charge: as given in `xtb_total_charge` (see *Protonation states*); electronic temperature 300 K
- Solvation: [TODO: gas phase / implicit solvent model]

## Dataset construction

- **Source:** TDC `AMES` dataset (7,255 compounds).
- **Exclusions:** 4 compounds are excluded because GFN2-xTB gave no usable result
  (hypervalent or non-convergent species):
  `Drug 6234` (`N#[N+]c1nc[nH]c1C(N)=O`), `Drug 6447` (`OCP(Cl)(CO)(CO)CO`),
  `Drug 7555` (`[O-][Cl+2]([O-])O`) and `Drug 864` (`[O-][Cl+3]([O-])([O-])[O-]`).
  No fallback or substitute values are used for any compound.
- **Composition:** 7,251 compounds (3,950 mutagenic, 3,301 non-mutagenic).
  There are 121,469 heavy atoms of 9 elements: C, N, O, F, P, S, Cl, Br and I.
- **Checks:** for every compound, the element and order of the heavy atoms in
  the xTB output match the RDKit atoms of `smiles`. The token walk finds the
  same number of atoms, and the token values sum back to the xTB atom values.

## Notes for users

- **Protonation states:** the xTB calculations do not always use the
  protonation state written in `smiles`. For 5,690 compounds the calculated species has
  the same net charge and hydrogen count as the SMILES. For 1,557 compounds
  it differs by protonation or deprotonation of ionisable groups, for example
  protonated amines (+1) and deprotonated carboxylic acids (−1), as expected
  for ionisation at physiological pH [TODO: confirm tool and pH]. For 4
  hypervalent phosphorus compounds (`[PH]` in the SMILES; `Drug 231`,
  `Drug 3258`, `Drug 3693`, `Drug 5482`), the calculated species has fewer
  hydrogens and a positive charge. In all 7,251 cases the heavy atoms and
  their order are identical. All descriptors describe the calculated
  species, and `xtb_total_charge` gives its net charge.
- **Near-degenerate frontier orbitals:** ω diverges as the HOMO–LUMO gap
  approaches zero. The smallest gap is 0.0051 Hartree, and 52 compounds have
  ω > 3 Hartree. These values are reported as computed. A log transform or a
  cap may help for some applications.
- **Open-shell compound:** `Drug 6896` is open-shell (odd electron count). Its
  `homo_Eh` is the energy of the singly occupied orbital.
- **`smiles` vs `canonical_smiles`:** the two columns are identical for 7,242
  compounds. For the other 9, they differ only in the placement of the `/`
  and `\` double-bond stereo markers, and describe the same molecule.
- **Standardisation:** all values are unstandardised. In the study, charges
  and atomic polarizabilities were used as is. HOMO, LUMO and ω were
  standardised with the dataset mean and SD (mean −0.38236, −0.26563,
  0.65388; SD 0.02813, 0.06543, 0.56662 Hartree).
- **Spreadsheet software:** some tokens (`=`, `+`, `-`) may be read as formulas
  when the file is opened directly in Excel. Import the `token` column as text,
  or use pandas or R.

## Loading

```python
import pandas as pd

mol = pd.read_csv("AMES_xTB_molecules.csv")
tok = pd.read_csv("AMES_xTB_tokens.csv", dtype={"atom_index": "Int64"})

# tokens and local descriptors for one compound
tok[tok.Drug_ID == "Drug 4860"]

# attach global descriptors to every token
tok_full = tok.merge(mol, on="Drug_ID")
```

## Code

The tables were generated with `scripts/build_ames_xtb_dataset.py` in
https://github.com/DinaKhasanova/physchem-xai. The script uses the same parsing
and token-mapping functions as model training
(`src/downstream/toxicity/features.py`).

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0).

The Ames labels and compound identifiers come from the Therapeutics Data Commons
(Huang et al., *NeurIPS Datasets and Benchmarks*, 2021). The TDC `AMES` set
is based on Xu et al., *J. Chem. Inf. Model.* 52, 2840–2847 (2012). Please cite
those sources along with this dataset.

## Citation

[TODO: manuscript citation]
[TODO: Zenodo DOI]
