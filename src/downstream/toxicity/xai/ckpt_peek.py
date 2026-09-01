"""Read a Lightning/PyTorch checkpoint without torch installed.

A ``.ckpt`` written by ``torch.save`` is a zip archive: a pickle describing the
tensors plus one raw byte blob per storage. Everything :mod:`diagnostics` needs
-- the two learned fusion scalars and two weight matrices -- can be recovered
with nothing but ``zipfile``, ``pickle`` and NumPy.

That matters when the machine holding the checkpoint has no GPU stack, which is
the normal situation when you just want to know whether training switched the
descriptor pathway off.

Usage
-----
    python -m downstream.toxicity.xai.ckpt_peek model.ckpt --molecule-features physchem
    python -m downstream.toxicity.xai.ckpt_peek model.ckpt --list
"""

from __future__ import annotations

import argparse
import pickle
import zipfile
from pathlib import Path

import numpy as np

# torch's legacy typed-storage class names -> numpy dtype
STORAGE_DTYPES: dict[str, np.dtype] = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("<u1"),
    "BoolStorage": np.dtype("?"),
}

MOLECULE_FEATURE_DIMS = {"physchem": 10, "orbital": 6}
MOLECULE_FEATURE_LABELS = {
    "physchem": (
        "MolWt", "TPSA", "LogP", "MolMR", "HBD", "HBA",
        "RotatableBonds", "RingCount", "HeavyAtomCount", "FractionCSP3",
    ),
    "orbital": (
        "HOMO", "LUMO", "HOMO_LUMO_gap",
        "chemical_potential", "hardness", "electrophilicity_index",
    ),
}


class _StorageRef:
    __slots__ = ("dtype", "key", "numel")

    def __init__(self, dtype: np.dtype | None, key: str, numel: int) -> None:
        self.dtype = dtype
        self.key = key
        self.numel = numel


class _LazyTensor:
    """Tensor metadata plus enough information to materialise it from the zip."""

    __slots__ = ("storage", "offset", "shape", "stride")

    def __init__(self, storage: _StorageRef, offset: int, shape: tuple, stride: tuple) -> None:
        self.storage = storage
        self.offset = offset
        self.shape = tuple(shape)
        self.stride = tuple(stride)

    def materialise(self, archive: zipfile.ZipFile, prefix: str) -> np.ndarray:
        if self.storage.dtype is None:
            raise NotImplementedError(
                "Unsupported storage dtype (bfloat16 or an untyped storage). "
                "Re-save the checkpoint in float32, or run diagnostics.py with torch."
            )
        raw = archive.read(f"{prefix}data/{self.storage.key}")
        flat = np.frombuffer(raw, dtype=self.storage.dtype)
        count = int(np.prod(self.shape)) if self.shape else 1
        window = flat[self.offset : self.offset + count]
        if window.size != count:
            raise ValueError(f"storage {self.storage.key} too small for shape {self.shape}")
        return window.reshape(self.shape)


class _Opaque:
    """Stand-in for any class in the pickle we do not care about."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state) -> None:
        self.state = state

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<opaque>"


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *_rest):
    return _LazyTensor(storage, int(storage_offset), tuple(size), tuple(stride))


def _identity(obj=None, *_args, **_kwargs):
    return obj


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name in STORAGE_DTYPES:
            return name  # persistent_id carries the class itself; the name is enough
        if name in {"UntypedStorage", "TypedStorage"}:
            return name
        if name in {"_rebuild_tensor_v2", "_rebuild_tensor", "_rebuild_tensor_v3"}:
            return _rebuild_tensor_v2
        if module.startswith("torch") or module.startswith("pytorch_lightning") or module.startswith("lightning"):
            return _Opaque
        if module == "collections" and name == "OrderedDict":
            import collections

            return collections.OrderedDict
        try:
            return super().find_class(module, name)
        except Exception:
            return _Opaque

    def persistent_load(self, pid):
        # ('storage', <storage class>, key, location, numel)
        if not isinstance(pid, tuple) or not pid or pid[0] != "storage":
            return _Opaque()
        _, storage_class, key, _location, numel = (list(pid) + [None] * 5)[:5]
        name = storage_class if isinstance(storage_class, str) else getattr(storage_class, "__name__", "")
        return _StorageRef(STORAGE_DTYPES.get(name), str(key), int(numel or 0))


def _find_prefix(archive: zipfile.ZipFile) -> str:
    for entry in archive.namelist():
        if entry.endswith("data.pkl"):
            return entry[: -len("data.pkl")]
    raise ValueError("Not a torch zip checkpoint: no data.pkl inside the archive.")


def load_tensors(checkpoint_path: str | Path) -> dict[str, np.ndarray]:
    """Return every tensor in the checkpoint's ``state_dict`` as a NumPy array."""

    path = Path(checkpoint_path)
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"{path} is not a zip-format checkpoint. Very old torch checkpoints "
            "use a different layout and need torch to read."
        )

    with zipfile.ZipFile(path) as archive:
        prefix = _find_prefix(archive)
        with archive.open(f"{prefix}data.pkl") as handle:
            payload = _Unpickler(handle).load()

        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError("Could not locate a state_dict inside the checkpoint.")

        tensors: dict[str, np.ndarray] = {}
        for key, value in state.items():
            if isinstance(value, _LazyTensor):
                try:
                    tensors[str(key)] = value.materialise(archive, prefix)
                except (NotImplementedError, ValueError, KeyError):
                    continue
        return tensors


# --------------------------------------------------------------------------
def analyse(
    tensors: dict[str, np.ndarray],
    token_feature_names: tuple[str, ...],
    molecule_feature_names: tuple[str, ...],
) -> str:
    lines: list[str] = []

    scales = {
        key.replace("model.", ""): float(np.asarray(value).reshape(-1)[0])
        for key, value in tensors.items()
        if key in {"model.charge_context_scale", "model.feature_stats_scale"}
    }
    if scales:
        lines.append("Learned descriptor-pathway scales (initialised at 0.1000 in finetune.py):")
        for name, value in scales.items():
            drift = abs(value - 0.1) / 0.1
            if abs(value) < 0.02:
                verdict = "COLLAPSED - pathway effectively switched off"
            elif drift < 0.01:
                verdict = "UNCHANGED from init - this parameter barely moved"
            elif value < 0.1:
                verdict = f"down-weighted ({drift:.0%} below init)"
            else:
                verdict = f"up-weighted ({drift:.0%} above init)"
            lines.append(f"  {name:<22} = {value: .4f}   {verdict}")
        lines.append("")
    else:
        lines.append("No fusion scales found - this looks like a SMILES-only checkpoint.")
        lines.append("")

    gain = tensors.get("model.classifier.0.weight")
    linear = tensors.get("model.classifier.1.weight")
    if gain is None or linear is None:
        lines.append("No ToxicityClassifier head found; nothing further to report.")
        return "\n".join(lines)

    in_features = int(gain.shape[0])
    encoder_hidden = int(linear.shape[0])
    token_dim = len(token_feature_names)
    molecule_dim = sum(MOLECULE_FEATURE_DIMS[name] for name in molecule_feature_names)
    stats_dim = token_dim * 4

    # Checkpoints trained with --descriptor-projection-dim route the descriptor
    # block through its own LayerNorm + Linear first, so the block that reaches
    # the classifier is the projection width, not the raw descriptor count.
    projection = tensors.get("model.descriptor_proj.1.weight")
    projected = projection is not None
    if projected:
        molecule_dim = int(projection.shape[0])

    expected = encoder_hidden * 2 + stats_dim + molecule_dim
    lines.append(f"Classifier input width: {in_features} (encoder hidden size {encoder_hidden})")
    if projected:
        lines.append(
            f"  descriptor block is PROJECTED: {int(projection.shape[1])} raw descriptors "
            f"-> {molecule_dim} dims, with its own LayerNorm"
        )
    if expected != in_features:
        lines.append(
            f"  Layout from the flags you passed would be {expected} - MISMATCH.\n"
            "  Pass the same --token-features/--molecule-features used for training."
        )
        leftover = in_features - encoder_hidden * 2
        lines.append(f"  Descriptor dimensions actually present: {leftover}")
        return "\n".join(lines)

    mass = np.abs(gain) * np.linalg.norm(linear, axis=0)

    # How far did the head actually move from its initialisation?
    #
    # nn.Linear seeds weights with kaiming_uniform_(a=sqrt(5)), i.e.
    # U(-1/sqrt(fan_in), 1/sqrt(fan_in)), whose per-column L2 norm over
    # out_features entries has expectation sqrt(out_features / (3 * fan_in)).
    # nn.LayerNorm seeds its gain at exactly 1.0. If the observed numbers sit on
    # top of those values, the head is still essentially its random projection
    # and whatever discrimination the model has was learned inside the encoder.
    init_mass = float(np.sqrt(encoder_hidden / (3.0 * in_features)))
    observed = float(mass.mean())
    gain_drift = float(np.abs(np.asarray(gain) - 1.0).mean())
    lines.append("How far is the head from its initialisation?")
    lines.append(f"  expected weight mass at initialisation : {init_mass:.4f}")
    lines.append(f"  observed mean weight mass             : {observed:.4f}  ({observed / init_mass:.2f}x init)")
    lines.append(f"  mean |LayerNorm gain - 1.0|           : {gain_drift:.4f}")
    if abs(observed / init_mass - 1.0) < 0.1 and gain_drift < 0.1:
        lines.append(
            "  => the weight magnitudes are at their initialisation values. This does not by\n"
            "     itself prove the head learned nothing (weights can rotate at constant norm),\n"
            "     but combined with a flat per-dimension profile below it means the head has\n"
            "     not learned to weight descriptor dimensions differently from encoder ones."
        )
    lines.append("")
    blocks: list[tuple[str, int, int]] = []
    start = 0
    for name, width in (
        ("cls", encoder_hidden),
        ("pooled_context", encoder_hidden),
        ("feature_stats", stats_dim),
        ("molecule_features", molecule_dim),
    ):
        if width:
            blocks.append((name, start, start + width))
        start += width

    total = float(mass.sum())
    lines.append("")
    lines.append("Per-input-dimension weight mass (|LayerNorm gain| x Linear column norm):")
    lines.append(f"  {'block':<18}{'width':>7}{'mean/dim':>12}{'total':>12}{'share':>9}")
    means: dict[str, float] = {}
    for name, lo, hi in blocks:
        block = mass[lo:hi]
        means[name] = float(block.mean())
        lines.append(
            f"  {name:<18}{hi - lo:>7}{block.mean():>12.4f}{block.sum():>12.2f}"
            f"{block.sum() / total:>8.1%}"
        )

    if "molecule_features" in means and "cls" in means and means["cls"] > 0:
        ratio = means["molecule_features"] / means["cls"]
        lines.append("")
        lines.append(f"  molecule_features carries {ratio:.2f}x the per-dimension mass of the CLS block.")
        if ratio < 1.0:
            lines.append(
                "  Below 1.0 means each descriptor dimension is weighted no more than an average\n"
                "  CLS dimension despite being far scarcer - the descriptor signal is diluted by\n"
                "  the shared LayerNorm over the whole concatenated vector."
            )

    if molecule_feature_names and not projected:
        labels: list[str] = []
        for name in molecule_feature_names:
            labels.extend(MOLECULE_FEATURE_LABELS[name])
        lo = encoder_hidden * 2 + stats_dim
        block = mass[lo : lo + len(labels)]
        if block.size == len(labels):
            lines.append("")
            lines.append("Per-descriptor weight mass (highest first):")
            for label, value in sorted(zip(labels, block), key=lambda item: -item[1]):
                lines.append(f"  {label:<24}{value:.4f}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--token-features", nargs="*", default=())
    parser.add_argument("--molecule-features", nargs="*", default=(), choices=("physchem", "orbital"))
    parser.add_argument("--list", action="store_true", help="Just list tensor names and shapes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensors = load_tensors(args.checkpoint)
    if args.list:
        for key in sorted(tensors):
            print(f"{key:<60}{tensors[key].shape}")
        return
    print(f"checkpoint: {args.checkpoint}")
    print(f"tensors read: {len(tensors)}")
    print()
    print(analyse(tensors, tuple(args.token_features), tuple(args.molecule_features)))


if __name__ == "__main__":
    main()
