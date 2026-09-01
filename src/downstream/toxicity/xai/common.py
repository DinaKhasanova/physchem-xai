"""Shared SMILES/token/atom alignment helpers.

The alignment here deliberately mirrors ``features._map_atom_values_to_smiles_characters``
character-for-character. If that function changes, change this one too, or the
attributions and the features will silently disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:  # torch is optional for the pure-string helpers below
    import torch
except Exception:  # pragma: no cover - exercised only in torch-less envs
    torch = None  # type: ignore[assignment]

UNSUPPORTED_TOKEN_CHARS = set("#()+-/12345678=@[]\\")


# --------------------------------------------------------------------------
# character <-> atom alignment
# --------------------------------------------------------------------------
def char_to_atom_index(smiles: str) -> list[int | None]:
    """For each character of ``smiles`` return the atom index it belongs to.

    Non-atom characters (bonds, ring closures, brackets, charge signs) map to
    ``None``. Multi-character atoms ("Cl", "Br", "[nH]") map every one of their
    characters to the same atom index, which is what makes it possible to sum
    character-level attributions back onto atoms without double counting.
    """

    mapping: list[int | None] = []
    atom_index = 0
    i = 0
    n = len(smiles)

    while i < n:
        ch = smiles[i]

        if ch == "[":
            end = smiles.find("]", i)
            if end == -1:
                mapping.append(None)
                i += 1
                continue
            for _ in range(end - i + 1):
                mapping.append(atom_index)
            atom_index += 1
            i = end + 1
            continue

        if ch == "C" and i + 1 < n and smiles[i + 1] == "l":
            mapping.extend([atom_index, atom_index])
            atom_index += 1
            i += 2
            continue

        if ch == "B" and i + 1 < n and smiles[i + 1] == "r":
            mapping.extend([atom_index, atom_index])
            atom_index += 1
            i += 2
            continue

        if ch in UNSUPPORTED_TOKEN_CHARS or not ch.isalpha():
            mapping.append(None)
            i += 1
            continue

        mapping.append(atom_index)
        atom_index += 1
        i += 1

    return mapping


def spread_atom_values_over_characters(smiles: str, atom_values: list[float]) -> list[float]:
    """Line-by-line mirror of ``features._map_atom_values_to_smiles_characters``.

    Written as the same walk rather than as a reimplementation on top of
    :func:`common.char_to_atom_index`, because any divergence here would put the
    perturbed molecule on a different feature scale from the training inputs and
    silently corrupt every counterfactual score. :mod:`selftest` asserts the two
    agree exactly.
    """

    values: list[float] = []
    atom_index = 0
    token_index = 0
    n = len(smiles)

    while token_index < n:
        token = smiles[token_index]

        if token == "[":
            end = smiles.find("]", token_index)
            if end == -1:
                values.append(0.0)
                token_index += 1
                continue

            value = atom_values[atom_index] if atom_index < len(atom_values) else 0.0
            atom_index += 1
            block = [0.0] * (end - token_index + 1)

            atom_pos = None
            for j in range(token_index + 1, end):
                if smiles[j].isalpha():
                    atom_pos = j
                    break
            if atom_pos is not None:
                block[atom_pos - token_index] = float(value)

            values.extend(block)
            token_index = end + 1
            continue

        if token == "C" and token_index + 1 < n and smiles[token_index + 1] == "l":
            value = atom_values[atom_index] if atom_index < len(atom_values) else 0.0
            values.extend([float(value) / 2.0, float(value) / 2.0])
            atom_index += 1
            token_index += 2
            continue

        if token == "B" and token_index + 1 < n and smiles[token_index + 1] == "r":
            value = atom_values[atom_index] if atom_index < len(atom_values) else 0.0
            values.extend([float(value) / 2.0, float(value) / 2.0])
            atom_index += 1
            token_index += 2
            continue

        if token in UNSUPPORTED_TOKEN_CHARS or not token.isalpha():
            values.append(0.0)
        else:
            value = atom_values[atom_index] if atom_index < len(atom_values) else 0.0
            values.append(float(value))
            atom_index += 1
        token_index += 1

    return values


def atom_to_char_indices(smiles: str) -> list[list[int]]:
    """Inverse of :func:`char_to_atom_index`: atom index -> character positions."""

    mapping = char_to_atom_index(smiles)
    n_atoms = 0
    for value in mapping:
        if value is not None:
            n_atoms = max(n_atoms, value + 1)
    spans: list[list[int]] = [[] for _ in range(n_atoms)]
    for char_index, atom_index in enumerate(mapping):
        if atom_index is not None:
            spans[atom_index].append(char_index)
    return spans


def parsed_atom_count(smiles: str) -> int:
    """Number of atoms the string walk believes are present."""

    mapping = char_to_atom_index(smiles)
    count = 0
    for value in mapping:
        if value is not None:
            count = max(count, value + 1)
    return count


def token_position_to_atom_index(smiles: str, token_position: int) -> int | None:
    """Map an *encoded* token position to an atom index.

    ``build_token_feature_matrix`` builds token vectors as
    ``[0.0] + one-value-per-SMILES-character + [0.0]``, so encoded position
    ``p`` corresponds to SMILES character ``p - 1``.
    """

    char_index = token_position - 1
    if char_index < 0 or char_index >= len(smiles):
        return None
    return char_to_atom_index(smiles)[char_index]


# --------------------------------------------------------------------------
# tokenizer sanity check
# --------------------------------------------------------------------------
@dataclass(slots=True)
class AlignmentReport:
    n_checked: int
    n_length_mismatch: int
    n_atom_count_mismatch: int
    examples: list[dict[str, object]]

    def ok(self) -> bool:
        return self.n_length_mismatch == 0 and self.n_atom_count_mismatch == 0

    def describe(self) -> str:
        lines = [
            "Token/character alignment report",
            f"  molecules checked          : {self.n_checked}",
            f"  tokenizer length mismatches: {self.n_length_mismatch}",
            f"  atom count mismatches      : {self.n_atom_count_mismatch}",
        ]
        if self.examples:
            lines.append("  first failures:")
            for example in self.examples[:10]:
                lines.append(f"    {example}")
        if self.ok():
            lines.append("  => alignment assumption HOLDS for this sample")
        else:
            lines.append(
                "  => alignment assumption BROKEN: token-level descriptors are "
                "being attached to the wrong positions for the listed molecules"
            )
        return "\n".join(lines)


def verify_token_alignment(encoder, smiles_list: list[str], rdkit_check: bool = True) -> AlignmentReport:
    """Check the two assumptions the token-feature pipeline silently relies on.

    1. the tokenizer emits exactly one token per SMILES character, plus one
       leading and one trailing special token;
    2. the string walk finds exactly as many atoms as RDKit does.

    Either assumption failing means every token-level physicochemical value is
    shifted or dropped, which would explain charge features not helping.
    """

    n_length_mismatch = 0
    n_atom_count_mismatch = 0
    examples: list[dict[str, object]] = []

    mol_from_smiles = None
    if rdkit_check:
        try:
            from rdkit import Chem  # noqa: PLC0415

            mol_from_smiles = Chem.MolFromSmiles
        except Exception:
            mol_from_smiles = None

    for smiles in smiles_list:
        encoded = encoder.tokenize([smiles])
        n_tokens = int(encoded["input_ids"].shape[1])
        expected = len(smiles) + 2
        if n_tokens != expected:
            n_length_mismatch += 1
            if len(examples) < 25:
                examples.append(
                    {
                        "smiles": smiles,
                        "issue": "token_length",
                        "n_tokens": n_tokens,
                        "expected": expected,
                    }
                )

        if mol_from_smiles is not None:
            mol = mol_from_smiles(smiles)
            if mol is not None:
                walked = parsed_atom_count(smiles)
                if walked != mol.GetNumAtoms():
                    n_atom_count_mismatch += 1
                    if len(examples) < 25:
                        examples.append(
                            {
                                "smiles": smiles,
                                "issue": "atom_count",
                                "walked": walked,
                                "rdkit": mol.GetNumAtoms(),
                            }
                        )

    return AlignmentReport(
        n_checked=len(smiles_list),
        n_length_mismatch=n_length_mismatch,
        n_atom_count_mismatch=n_atom_count_mismatch,
        examples=examples,
    )


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def resolve_device():
    """Pick cuda > mps > cpu, matching the convention used elsewhere in the repo."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for resolve_device()")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


BLOCK_ORDER = ("cls", "pooled_context", "feature_stats", "molecule_features")


def classifier_block_slices(
    encoder_hidden_size: int,
    token_feature_dim: int,
    molecule_feature_dim: int,
) -> dict[str, slice]:
    """Column ranges of ``ToxicityClassifier.classifier[1]`` input.

    The classifier input is ``cat([cls, pooled_context, feature_stats, molecule_features])``
    (see ``finetune.ToxicityClassifier.forward``).
    """

    stats_dim = token_feature_dim * 4 if token_feature_dim > 0 else 0
    start = 0
    slices: dict[str, slice] = {}
    for name, width in (
        ("cls", encoder_hidden_size),
        ("pooled_context", encoder_hidden_size),
        ("feature_stats", stats_dim),
        ("molecule_features", molecule_feature_dim),
    ):
        slices[name] = slice(start, start + width)
        start += width
    return slices


# --------------------------------------------------------------------------
# small statistics helpers (kept dependency-free so they are testable anywhere)
# --------------------------------------------------------------------------
def curve_auc(values: list[float]) -> float:
    if len(values) < 2:
        return float(values[0]) if values else 0.0
    total = 0.0
    for i in range(len(values) - 1):
        total += (values[i] + values[i + 1]) / 2.0
    return total / (len(values) - 1)


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(a) != len(b):
        return float("nan")

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    ra, rb = rank(a), rank(b)
    n = len(ra)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    den_a = sum((ra[i] - mean_a) ** 2 for i in range(n)) ** 0.5
    den_b = sum((rb[i] - mean_b) ** 2 for i in range(n)) ** 0.5
    if den_a == 0 or den_b == 0:
        return float("nan")
    return num / (den_a * den_b)
