"""Self-tests for the XAI package.

Split into two tiers so it is useful in both environments:

* ``--pure`` runs only tests with no torch / RDKit dependency. These pass in any
  Python 3.10+ environment and cover the string-walk logic that everything else
  is built on.
* the default run additionally checks, against the real ``features`` module,
  that :func:`counterfactual_occlusion.token_matrix_from_atom_values` reproduces
  ``features.build_token_feature_matrix`` exactly. If that equivalence breaks,
  the counterfactual attributions are on a different scale from training and
  the numbers are not comparable.

Usage
-----
    python -m downstream.toxicity.xai.selftest --pure       # anywhere
    python -m downstream.toxicity.xai.selftest              # on the GPU VM
"""

from __future__ import annotations

import argparse
import sys

PASS = "PASS"
FAIL = "FAIL"


class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.rows.append((PASS if condition else FAIL, name, detail))

    def report(self) -> int:
        width = max(len(name) for _, name, _ in self.rows) if self.rows else 10
        failures = 0
        for status, name, detail in self.rows:
            marker = " " if status == PASS else ">"
            print(f"{marker} {status}  {name:<{width}}  {detail}")
            if status == FAIL:
                failures += 1
        print()
        print(f"{len(self.rows) - failures} passed, {failures} failed")
        return failures


# --------------------------------------------------------------------------
def _ckpt_peek_test(results: Results) -> None:
    """Round-trip a synthetic checkpoint written in torch's on-disk zip format.

    Skipped when real torch is importable, because the writer has to install
    stub ``torch`` modules for the pickler and we will not shadow the real one.
    """

    import collections
    import io
    import pickle
    import sys
    import tempfile
    import types
    import zipfile

    try:
        import numpy as np
    except ImportError:
        results.check("ckpt_peek round trip", False, "skipped: numpy missing")
        return

    if "torch" in sys.modules:
        results.check("ckpt_peek round trip", True, "skipped: real torch present, use diagnostics.py")
        return

    saved = {name: sys.modules.get(name) for name in ("torch", "torch._utils")}
    try:
        torch_stub = types.ModuleType("torch")
        utils_stub = types.ModuleType("torch._utils")
        torch_stub.__spec__ = utils_stub.__spec__ = None
        sys.modules["torch"] = torch_stub
        sys.modules["torch._utils"] = utils_stub

        class FloatStorage:
            pass

        FloatStorage.__module__ = "torch"
        FloatStorage.__qualname__ = "FloatStorage"
        torch_stub.FloatStorage = FloatStorage

        def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks):
            ...

        _rebuild_tensor_v2.__module__ = "torch._utils"
        _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
        utils_stub._rebuild_tensor_v2 = _rebuild_tensor_v2

        blobs: dict[str, bytes] = {}

        class Ref:
            def __init__(self, key, numel):
                self.key, self.numel = key, numel

        class FakeTensor:
            def __init__(self, array, key):
                array = np.ascontiguousarray(array, dtype="<f4")
                blobs[key] = array.tobytes()
                self.ref, self.shape = Ref(key, array.size), array.shape

            def __reduce__(self):
                stride = tuple(int(np.prod(self.shape[i + 1 :])) for i in range(len(self.shape)))
                return (
                    _rebuild_tensor_v2,
                    (self.ref, 0, self.shape, stride, False, collections.OrderedDict()),
                )

        encoder_hidden, token_dim, molecule_dim = 8, 1, 10
        width = encoder_hidden * 2 + token_dim * 4 + molecule_dim
        rng = np.random.default_rng(0)
        gain = rng.normal(1.0, 0.1, width)
        linear = rng.normal(0.0, 0.05, (encoder_hidden, width))

        payload = {
            "epoch": 3,
            "state_dict": collections.OrderedDict(
                [
                    ("model.charge_context_scale", FakeTensor(np.array([0.0031]), "0")),
                    ("model.feature_stats_scale", FakeTensor(np.array([0.24]), "1")),
                    ("model.classifier.0.weight", FakeTensor(gain, "2")),
                    ("model.classifier.1.weight", FakeTensor(linear, "3")),
                ]
            ),
        }

        class TorchPickler(pickle.Pickler):
            def persistent_id(self, obj):
                if isinstance(obj, Ref):
                    return ("storage", FloatStorage, obj.key, "cpu", obj.numel)
                return None

        buffer = io.BytesIO()
        TorchPickler(buffer, protocol=2).dump(payload)

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as handle:
            path = handle.name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("archive/data.pkl", buffer.getvalue())
            for key, blob in blobs.items():
                archive.writestr(f"archive/data/{key}", blob)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    from .ckpt_peek import analyse, load_tensors

    tensors = load_tensors(path)
    results.check(
        "ckpt_peek reads tensors without torch",
        set(tensors) == set(payload["state_dict"]),
        f"{len(tensors)} tensors recovered",
    )
    results.check(
        "ckpt_peek values are byte exact",
        np.allclose(tensors["model.classifier.0.weight"], gain, atol=1e-6)
        and np.allclose(tensors["model.classifier.1.weight"], linear, atol=1e-6)
        and abs(float(tensors["model.charge_context_scale"][0]) - 0.0031) < 1e-7,
    )
    report = analyse(tensors, ("charge",), ("physchem",))
    results.check(
        "ckpt_peek flags a collapsed scale",
        "COLLAPSED" in report and "MolWt" in report,
    )


def pure_tests(results: Results) -> None:
    from .common import (
        atom_to_char_indices,
        char_to_atom_index,
        curve_auc as _curve_auc,
        parsed_atom_count,
        spearman as _spearman,
    )
    from .wins import exact_mcnemar

    # simple chain
    results.check(
        "char_to_atom_index/CCO",
        char_to_atom_index("CCO") == [0, 1, 2],
        "three single-character atoms",
    )
    # ring closure digits and bonds are not atoms
    results.check(
        "char_to_atom_index/c1ccccc1",
        char_to_atom_index("c1ccccc1") == [0, None, 1, 2, 3, 4, 5, None],
        "digits map to None",
    )
    # two-character halogens share one atom index
    results.check(
        "char_to_atom_index/CCl",
        char_to_atom_index("CCl") == [0, 1, 1],
        "Cl occupies two characters, one atom",
    )
    results.check(
        "char_to_atom_index/CBr",
        char_to_atom_index("CBr") == [0, 1, 1],
        "Br occupies two characters, one atom",
    )
    # bracket atoms
    results.check(
        "char_to_atom_index/[nH]",
        char_to_atom_index("c1cc[nH]c1") == [0, None, 1, 2, 3, 3, 3, 3, 4, None],
        "the whole bracket belongs to one atom",
    )
    results.check(
        "char_to_atom_index/[N+]",
        char_to_atom_index("C[N+](=O)[O-]")
        == [0, 1, 1, 1, 1, None, None, 2, None, 3, 3, 3, 3],
        "charged bracket atoms",
    )

    from .common import spread_atom_values_over_characters as spread

    results.check("spread/CCO", spread("CCO", [1, 2, 3]) == [1.0, 2.0, 3.0])
    results.check(
        "spread/CCl halves the value",
        spread("CCl", [1, 2]) == [1.0, 1.0, 1.0],
        "Cl's 2.0 is split across 'C' and 'l'",
    )
    results.check(
        "spread/bracket puts value on the element symbol",
        spread("C[N+](=O)[O-]", [1, 2, 3, 4])
        == [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0],
    )
    results.check(
        "spread/[nH]",
        spread("c1cc[nH]c1", [1, 2, 3, 4, 5]) == [1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0],
    )
    results.check(
        "spread/length equals len(smiles)",
        all(len(spread(s, list(range(20)))) == len(s) for s in ["CCO", "CCl", "c1ccccc1N", "C[N+](=O)[O-]"]),
    )
    results.check(
        "spread/missing values default to zero",
        spread("CCO", []) == [0.0, 0.0, 0.0],
    )

    results.check("parsed_atom_count/CCO", parsed_atom_count("CCO") == 3)
    results.check("parsed_atom_count/CCl", parsed_atom_count("CCl") == 2)
    results.check(
        "atom_to_char_indices/CCl",
        atom_to_char_indices("CCl") == [[0], [1, 2]],
    )

    # spearman
    results.check("spearman/monotone", abs(_spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9)
    results.check("spearman/reversed", abs(_spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9)
    results.check("spearman/short", _spearman([1, 2], [1, 2]) != _spearman([1, 2], [1, 2]), "NaN for n<3")

    # curve auc
    results.check("curve_auc/flat", abs(_curve_auc([0.5, 0.5, 0.5]) - 0.5) < 1e-9)
    results.check("curve_auc/linear", abs(_curve_auc([0.0, 1.0]) - 0.5) < 1e-9)
    results.check("curve_auc/single", abs(_curve_auc([0.7]) - 0.7) < 1e-9)

    _ckpt_peek_test(results)

    # mcnemar
    results.check("mcnemar/no_discordance", abs(exact_mcnemar(0, 0) - 1.0) < 1e-12)
    results.check("mcnemar/symmetric", abs(exact_mcnemar(5, 5) - 1.0) < 1e-9)
    results.check("mcnemar/strong", exact_mcnemar(0, 12) < 0.001, f"p={exact_mcnemar(0, 12):.2g}")
    results.check("mcnemar/monotone", exact_mcnemar(2, 12) < exact_mcnemar(5, 9))


# --------------------------------------------------------------------------
def integration_tests(results: Results) -> None:
    import torch

    from ..features import build_token_feature_matrix
    from .common import char_to_atom_index
    from .counterfactual_occlusion import (
        delete_atom,
        parent_atom_values,
        token_matrix_from_atom_values,
    )

    samples = [
        "CCO",
        "c1ccccc1N",
        "CCl",
        "CBr",
        "C[N+](=O)[O-]",
        "c1cc[nH]c1",
        "Cc1ccc(N)cc1Cl",
        "O=C(O)c1ccccc1O",
    ]
    feature_sets = [("charge",), ("neighbor_k3",), ("charge", "polarizability", "neighbor_k2")]

    worst = 0.0
    mismatched: list[str] = []
    for smiles in samples:
        for names in feature_sets:
            length = len(smiles) + 2
            reference = build_token_feature_matrix(
                smiles=smiles,
                feature_names=names,
                encoded_length=length,
            )
            values = parent_atom_values(smiles, names, None, None)
            candidate = token_matrix_from_atom_values(
                smiles=smiles,
                feature_names=names,
                atom_values=values,
                encoded_length=length,
            )
            if reference.shape != candidate.shape:
                mismatched.append(f"{smiles}/{names}: shape {tuple(reference.shape)} vs {tuple(candidate.shape)}")
                continue
            delta = float((reference - candidate).abs().max().item())
            worst = max(worst, delta)
            if delta > 1e-5:
                mismatched.append(f"{smiles}/{names}: max abs diff {delta:.2e}")

    results.check(
        "token_matrix_from_atom_values == build_token_feature_matrix",
        not mismatched,
        f"worst abs diff {worst:.2e}" if not mismatched else "; ".join(mismatched[:4]),
    )

    # atom deletion round trip
    from rdkit import Chem

    ok = True
    details: list[str] = []
    for smiles in ["Cc1ccc(N)cc1Cl", "O=C(O)c1ccccc1O", "C[N+](=O)[O-]"]:
        mol = Chem.MolFromSmiles(smiles)
        for atom_index in range(mol.GetNumAtoms()):
            new_smiles, parent_positions = delete_atom(smiles, atom_index)
            if new_smiles is None:
                continue
            new_mol = Chem.MolFromSmiles(new_smiles)
            if new_mol is None:
                ok = False
                details.append(f"{smiles} -{atom_index} gave unparseable {new_smiles}")
                continue
            walked = len([v for v in set(char_to_atom_index(new_smiles)) if v is not None])
            if walked != new_mol.GetNumAtoms():
                ok = False
                details.append(f"{new_smiles}: walk {walked} vs rdkit {new_mol.GetNumAtoms()}")
            if len(parent_positions) != new_mol.GetNumAtoms():
                ok = False
                details.append(f"{new_smiles}: {len(parent_positions)} parent positions for {new_mol.GetNumAtoms()} atoms")
    results.check("delete_atom round trip", ok, "; ".join(details[:4]) or "all deletions valid")

    # element identity is preserved when carrying parent values across a deletion
    ok = True
    details = []
    for smiles in ["Cc1ccc(N)cc1Cl", "O=C(O)c1ccccc1O"]:
        mol = Chem.MolFromSmiles(smiles)
        parent_elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for atom_index in range(mol.GetNumAtoms()):
            new_smiles, parent_positions = delete_atom(smiles, atom_index)
            if new_smiles is None:
                continue
            new_mol = Chem.MolFromSmiles(new_smiles)
            Chem.MolToSmiles(new_mol, canonical=True)
            from .counterfactual_occlusion import _smiles_output_order

            order = _smiles_output_order(new_mol)
            new_elements = [new_mol.GetAtomWithIdx(i).GetSymbol() for i in order]
            carried = [parent_elements[p] for p in parent_positions]
            if new_elements != carried:
                ok = False
                details.append(f"{smiles} -{atom_index}: {new_elements} vs {carried}")
    results.check(
        "parent value carry-over keeps elements aligned",
        ok,
        "; ".join(details[:3]) or "element sequences match after every deletion",
    )

    results.check("torch import", torch.__version__ is not None, f"torch {torch.__version__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pure", action="store_true", help="Skip tests needing torch/RDKit")
    args = parser.parse_args()

    results = Results()
    pure_tests(results)
    if not args.pure:
        try:
            integration_tests(results)
        except ImportError as exc:
            results.check("integration tests", False, f"skipped: {exc}")

    sys.exit(1 if results.report() else 0)


if __name__ == "__main__":
    main()
