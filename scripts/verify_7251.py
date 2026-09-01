"""Pre-flight verification before the full table run.

1. dataset size and index coverage  -> exactly 7,251, zero misses
2. assignment correctness           -> element multiset of each .out matches
                                       the molecule's own heavy atoms
3. token alignment                  -> descriptors land on atom characters
                                       only; CLS/SEP/padding stay zero
"""
import re, sys
from pathlib import Path
import pandas as pd
from rdkit import Chem

from downstream.toxicity.features import (
    build_token_feature_matrix, _build_xyz_index, _canonical_smiles,
)
from molencoder.model import MolEncoderWrapper
from molencoder.config import MolEncoderConfig

DATA = "ames_7251.csv"
XYZ = "xtb_7251"

df = pd.read_csv(DATA)
print("=" * 70); print("1. DATASET AND INDEX"); print("=" * 70)
print(f"rows in {DATA}                : {len(df)}")
print(f"unique SMILES                 : {df.smiles.nunique()}")
idx = _build_xyz_index(Path(XYZ))
print(f"index entries                 : {len(idx)}")
miss = [s for s in df.smiles if _canonical_smiles(s) not in idx]
print(f"molecules with NO .out file   : {len(miss)}   <-- must be 0")
print(f".out files on disk            : {len(list(Path(XYZ,'outs').glob('*.out')))}")

print(); print("=" * 70)
print("2. ASSIGNMENT CORRECTNESS  (element multiset: molecule vs .out)")
print("=" * 70)
ELEM = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)")

def elements_in_out(path):
    els, seen = [], False
    for line in Path(path).read_text(errors="replace").splitlines():
        m = ELEM.match(line)
        if m:
            els.append(m.group(1)); seen = True
        elif seen and els and not line.strip():
            break
    return els

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ok = bad = unparsed = 0
examples = []
for s in df.smiles.head(N):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        unparsed += 1; continue
    key = _canonical_smiles(s)
    if key not in idx:
        bad += 1; continue
    mol_els = sorted(a.GetSymbol() for a in mol.GetAtoms())
    out_els = sorted(e for e in elements_in_out(idx[key]) if e != "H")
    if mol_els == out_els:
        ok += 1
    else:
        bad += 1
        if len(examples) < 5:
            examples.append((s, "".join(mol_els[:14]), "".join(out_els[:14])))
print(f"checked {N}:  match {ok}   MISMATCH {bad}   unparseable {unparsed}")
for s, a, b in examples:
    print(f"   {s[:40]:40s} mol={a}  out={b}")

print(); print("=" * 70)
print("3. TOKEN ALIGNMENT  (atoms only; CLS/SEP/pad must be zero)")
print("=" * 70)
enc = MolEncoderWrapper(MolEncoderConfig(
    pretrained_model_name="fabikru/MolEncoder", max_sequence_length=128,
    num_labels=2, model_cache_dir="models/molencoder"))
tok = enc.tokenizer
NONATOM = set("()[]=#@+-/\\.%0123456789")
cls_nz = sep_nz = pad_nz = nonatom_nz = checked = 0
for s in df.smiles.head(150):
    ids = tok(s, padding="max_length", truncation=True,
              max_length=128, return_tensors="pt")["input_ids"][0]
    v = build_token_feature_matrix(s, ["charge"], 128,
                                   normalize_mode="none",
                                   charge_xyz_dir=XYZ).flatten()
    for t, x in zip(tok.convert_ids_to_tokens(ids.tolist()), v.tolist()):
        if t in (tok.cls_token, tok.bos_token) and x != 0: cls_nz += 1
        elif t in (tok.sep_token, tok.eos_token) and x != 0: sep_nz += 1
        elif t == tok.pad_token and x != 0: pad_nz += 1
        elif len(t) == 1 and t in NONATOM and x != 0: nonatom_nz += 1
    checked += 1
print(f"checked {checked} molecules")
print(f"  CLS token non-zero          : {cls_nz}   <-- must be 0")
print(f"  SEP/EOS token non-zero      : {sep_nz}   <-- must be 0")
print(f"  PAD token non-zero          : {pad_nz}   <-- must be 0")
print(f"  bond/bracket/digit non-zero : {nonatom_nz}   <-- must be 0")
print(); print("=" * 70)
print("VERDICT: proceed only if every '<-- must be 0' line reads 0")
print("=" * 70)
