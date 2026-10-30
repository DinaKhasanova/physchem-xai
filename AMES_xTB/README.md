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


## Molecule-level table: `AMES_xTB_molecules.csv`

| Column | Type | Unit | Description |
|---|---|---|---|
| `Drug_ID` | string | | TDC compound identifier (primary key) |
| `smiles` | string | | smiles string that was tokenised |
| `canonical_smiles` | string | | RDKit canonical smiles|
| `ames_label` | int | | Ames test result: 1 = mutagenic, 0 = non-mutagenic |
| `xtb_output_file` | string | | name of the GFN2-xTB output file (in `data/xtb_7251/outs/` of the GitHub repository) |
| `xtb_total_charge` | int | e | total molecular charge of the xTB calculation (see *Protonation states*) |
| `n_heavy_atoms` | int | | number of non-hydrogen atoms |
| `n_atoms_total` | int | | number of atoms, including hydrogens |
| `n_tokens` | int | | number of SMILES tokens |
| `homo_Eh` | float | Hartree | energy of the highest occupied molecular orbital |
| `lumo_Eh` | float | Hartree | energy of the lowest unoccupied molecular orbital |
| `homo_lumo_gap_Eh` | float | Hartree | LUMO − HOMO |
| `electrophilicity_index_Eh` | float | Hartree | ω = μ² / (2η), with μ = (E_HOMO + E_LUMO)/2 and η = E_LUMO − E_HOMO|
| `total_energy_Eh` | float | Hartree | GFN2-xTB total energy of the single-point calculation |
| `total_polarizability_bohr3` | float | bohr³ (a.u.) | molecular static dipole polarizability (the sum of the atomic polarizabilities of all atoms, including hydrogens) |


## Token-level table: `AMES_xTB_tokens.csv`

| Column | Type | Unit | Description |
|---|---|---|---|
| `Drug_ID` | string | | TDC compound identifier (links to the molecule table) |
| `token_index` | int | | 0-based position of the token in `smiles` |
| `token` | string | | SMILES character |
| `atom_index` | int or empty | | index of the heavy atom the token belongs to |
| `element` | string or empty | | element symbol of the heavy atom |
| `xtb_charge_e` | float | e | GFN2-xTB atomic partial charge mapped onto the token |
| `xtb_polarizability_bohr3` | float | bohr³ (a.u.) | GFN2-xTB atomic static dipole polarizability |



### How descriptor values are mapped onto tokens


- **Single-letter atoms** (`C`, `c`, `N` ...) carry the full value of their atom.
- **Two-letter halogens** (`Cl`, `Br`) are two tokens. The
  atom's value is split equally between them, so each token carries half.
- **Bracket atoms** (for example `[nH]`, `[N+]`, `[C@@H]`): the bracket
  describes one atom, including its hydrogen count, charge and chirality, so
  every character of the bracket has the same `atom_index`. The full value is
  on the first letter of the element symbol, and the other characters in the
  bracket carry 0.0.
- **Other tokens** (bonds `- = # / \`, branches `( )`, ring-closure digits)
  carry 0.0 and have an empty `atom_index`.


The model input also contains a `[CLS]` token at the start and a `[SEP]` token
at the end. Both carry zeros and are not listed, so the model position of a
token is `token_index + 1`.


## Calculation details

GFN2-xTB calculations were performed with ULYSSES. Three-dimensional structures
were generated from SMILES with CORINA Classic (MN-AM), protonation states were
assigned with Schrodinger Epik at pH 7.4 +/- 0.2, and the geometries were optimised
with GFN2-xTB before the single-point calculation.
The TDC `AMES` set contains 7,255 compounds. Four are excluded because protonation
state assignment (Schrodinger Epik, pH 7.4 +/- 0.2) returned no usable structure for
them, so they never entered the GFN2-xTB step: `Drug 6234`, `Drug 6447`, `Drug 7555`
and `Drug 864`.

**Protonation states:** for 1,561 compounds, the calculated species is in a
different protonation state from the one written in `smiles`, for example
protonated amines (+1) or deprotonated carboxylic acids (−1). In 4 hypervalent
phosphorus compounds written with `[PH]`, the calculated species has fewer
hydrogens. In all 7,251 compounds the heavy atoms and their order are
identical. All descriptors refer to the calculated species, whose net charge
is given in `xtb_total_charge`.

## Code

The tables were generated with `scripts/build_ames_xtb_dataset.py` in
https://github.com/DinaKhasanova/physchem-xai. The script uses the same parsing
and token-mapping functions as model training (`src/downstream/toxicity/features.py`).

## Licence

The data in this record are released under the Creative Commons Attribution
4.0 International licence (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/). The code used to generate them
(https://github.com/DinaKhasanova/physchem-xai) is released under the MIT licence.


## Citation

If you use this dataset, please cite it by its DOI:
https://doi.org/10.5281/zenodo.22919496

The accompanying manuscript is in preparation.
