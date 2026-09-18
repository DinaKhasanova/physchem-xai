#!/usr/bin/env python
"""
    python draw_ig_panels.py \\
        --ig-a results/xai/ig_baseline.csv --label-a "SMILES-only model" \\
        --ig-b results/xai/ig_gasteiger.csv --label-b "SMILES + Gasteiger charges" \\
        --smiles "COc1ccccc1[N+](=O)[O-]" "CS(=O)(=O)OCCCCOS(C)(=O)=O" \\
        --out figures/panels.html

the n largest prediction shifts:

    python draw_ig_panels.py --ig-a A.csv --ig-b B.csv \\
        --top-shift n --cohort results/xai/flip_cohort.csv \\
        --out figures/top.html
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

REQUIRED_COLUMNS = {"smiles", "fold", "atom_index", "attr_token",
                    "attr_descriptor", "attr_total"}

POSITIVE_RGB = (0.90, 0.20, 0.20)
NEGATIVE_RGB = (0.20, 0.75, 0.30)


def draw_panel(smiles, values, scale, title, width=520, height=420,
               decimals=2, annotate=True, highlight_radius=0.42):
    """Return an HTML fragment: one molecule, atoms coloured and labelled.

    values   per-atom attributions, index-aligned with RDKit atom order
    scale    value mapped to full colour saturation
    decimals digits shown beside each atom
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"<div style='color:#a00'>unparseable: {html.escape(smiles)}</div>"
    AllChem.Compute2DCoords(mol)

    n = min(len(values), mol.GetNumAtoms())
    if annotate:
        for i in range(n):
            mol.GetAtomWithIdx(i).SetProp(
                "atomNote", f"{float(values[i]):+.{decimals}f}")

    highlights, colors, radii = [], {}, {}
    for i in range(n):
        v = float(values[i])
        if abs(v) < 1e-9 or scale <= 0:
            continue
        frac = max(0.0, min(1.0, abs(v) / scale))
        if frac < 0.02:
            continue
        base = POSITIVE_RGB if v > 0 else NEGATIVE_RGB
        colors[i] = tuple(max(0.0, min(1.0, 1.0 - frac * (1.0 - c)))
                          for c in base)
        highlights.append(i)
        radii[i] = highlight_radius

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    if annotate:
        opts.annotationFontScale = 0.75
        opts.addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=highlights,
        highlightAtomColors=colors, highlightAtomRadii=radii)
    drawer.FinishDrawing()

    lo, hi = (min(values[:n]), max(values[:n])) if n else (0.0, 0.0)
    return (
        "<div style='text-align:center;padding:8px'>"
        f"<div style='font:600 13px system-ui;margin-bottom:4px'>"
        f"{html.escape(title)}</div>"
        f"{drawer.GetDrawingText()}"
        f"<div style='font:11px ui-monospace,monospace;color:#666;margin-top:2px'>"
        f"range {lo:+.3f} … {hi:+.3f} &nbsp;|&nbsp; scale ±{scale:.3f}</div>"
        "</div>")


def atom_table(smiles, a_tot, b_tok, b_desc, b_tot, label_a, label_b,
               mark_top=3):
    """Return an HTML table of per-atom values and the A→B change."""
    mol = Chem.MolFromSmiles(smiles)
    syms = [a.GetSymbol() for a in mol.GetAtoms()] if mol else []
    n = min(len(a_tot), len(b_tot), len(syms) or len(a_tot))

    def cell(v, bold=False, signed=True):
        color = "#b00" if v > 1e-9 else ("#070" if v < -1e-9 else "#999")
        text = f"{v:+.3f}" if signed else f"{v:.3f}"
        return (f"<td style='text-align:right;padding:2px 8px;color:{color};"
                f"font-weight:{'600' if bold else '400'}'>{text}</td>")

    deltas = [abs(b_tot[i] - a_tot[i]) for i in range(n)]
    top = set(sorted(range(n), key=lambda i: -deltas[i])[:mark_top])

    rows = []
    for i in range(n):
        rows.append(
            "<tr>"
            f"<td style='padding:2px 8px;color:#444'>{i}</td>"
            f"<td style='padding:2px 8px;font-weight:600'>"
            f"{syms[i] if i < len(syms) else '?'}</td>"
            + cell(a_tot[i]) + cell(b_tot[i])
            + cell(b_tot[i] - a_tot[i], bold=True)
            + cell(b_tok[i]) + cell(b_desc[i])
            + f"<td style='color:#c60;padding:2px 4px'>"
              f"{'&#9664;' if i in top else ''}</td>"
            "</tr>")

    header = (
        "<tr style='border-bottom:1px solid #ccc'>"
        "<th style='padding:3px 8px;text-align:left'>#</th>"
        "<th style='padding:3px 8px;text-align:left'>atom</th>"
        f"<th style='padding:3px 8px;text-align:right'>{html.escape(label_a)}<br>total</th>"
        f"<th style='padding:3px 8px;text-align:right'>{html.escape(label_b)}<br>total</th>"
        "<th style='padding:3px 8px;text-align:right'>change</th>"
        "<th style='padding:3px 8px;text-align:right'>token<br>pathway</th>"
        "<th style='padding:3px 8px;text-align:right'>descriptor<br>pathway</th>"
        "<th></th></tr>")

    totals = ("<tr style='border-top:1px solid #ccc;font-weight:600'>"
              "<td colspan='2' style='padding:3px 8px'>&Sigma;|attr|</td>"
              + cell(sum(abs(v) for v in a_tot[:n]), signed=False)
              + cell(sum(abs(v) for v in b_tot[:n]), signed=False)
              + "<td></td>"
              + cell(sum(abs(v) for v in b_tok[:n]), signed=False)
              + cell(sum(abs(v) for v in b_desc[:n]), signed=False)
              + "<td></td></tr>")

    return ("<div style='margin:10px auto;max-width:860px'>"
            "<table style='border-collapse:collapse;font:12px ui-monospace,"
            "SFMono-Regular,Menlo,monospace;margin:0 auto'>"
            f"{header}{''.join(rows)}{totals}</table>"
            "<div style='font:11px system-ui;color:#777;text-align:center;"
            "margin-top:4px'>&#9664; marks the atoms whose total attribution "
            "changed most. The token and descriptor pathways sum to the "
            "second model's total.</div></div>")


def load_ig(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing required column(s): {sorted(missing)}")
    return df


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def resolve_smiles(requested, available):
    """Match requested SMILES against the CSV's spellings, canonically."""
    lookup = {}
    for s in available:
        can = canonical(s)
        if can:
            lookup.setdefault(can, s)
    resolved, missing = [], []
    for s in requested:
        can = canonical(s)
        if can is None:
            missing.append(f"{s} (unparseable)")
        elif can in lookup:
            resolved.append(lookup[can])
        else:
            missing.append(s)
    return resolved, missing


def select_by_shift(cohort_path: Path, available, n: int):
    """Top-n molecules by |prob_a - prob_b| from a flip-cohort style CSV."""
    c = pd.read_csv(cohort_path)
    need = {"smiles", "prob_baseline", "prob_descriptor"}
    if not need <= set(c.columns):
        sys.exit(f"{cohort_path}: --top-shift needs columns {sorted(need)}")
    c = c.assign(shift=(c.prob_baseline - c.prob_descriptor).abs())
    ranked = c.sort_values("shift", ascending=False).smiles.tolist()
    resolved, _ = resolve_smiles(ranked, available)
    return resolved[:n]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ig-a", required=True, type=Path,
                   help="attribution CSV for the reference model")
    p.add_argument("--ig-b", required=True, type=Path,
                   help="attribution CSV for the comparison model")
    p.add_argument("--label-a", default="model A")
    p.add_argument("--label-b", default="model B")
    p.add_argument("--out", required=True, type=Path,
                   help="output HTML path")

    src = p.add_argument_group("molecule selection (choose one)")
    src.add_argument("--smiles", nargs="+", metavar="SMILES",
                     help="molecules to draw; matched canonically")
    src.add_argument("--smiles-file", type=Path,
                     help="file with one SMILES per line; an optional "
                          "tab-separated second field is used as the caption")
    src.add_argument("--top-shift", type=int, metavar="N",
                     help="draw the N molecules with the largest prediction "
                          "shift (requires --cohort)")
    src.add_argument("--cohort", type=Path,
                     help="cohort CSV with prob_baseline / prob_descriptor")

    p.add_argument("--fold", type=int,
                   help="restrict to one fold (default: lowest available)")
    p.add_argument("--strict", action="store_true",
                   help="exit with an error if any requested molecule cannot "
                        "be drawn, instead of skipping it")
    p.add_argument("--no-table", action="store_true",
                   help="structures only, no per-atom table")
    p.add_argument("--no-pathways", action="store_true",
                   help="omit the token/descriptor pathway panels")
    p.add_argument("--width", type=int, default=520)
    p.add_argument("--height", type=int, default=420)
    p.add_argument("--title", default="Integrated-gradients attributions")
    args = p.parse_args()

    if sum(x is not None for x in (args.smiles, args.smiles_file,
                                   args.top_shift)) != 1:
        p.error("choose exactly one of --smiles, --smiles-file, --top-shift")
    if args.top_shift is not None and args.cohort is None:
        p.error("--top-shift requires --cohort")

    a_df, b_df = load_ig(args.ig_a), load_ig(args.ig_b)
    available = sorted(set(a_df.smiles) & set(b_df.smiles))
    if not available:
        sys.exit("no molecules present in both attribution files")

    captions = {}
    if args.smiles is not None:
        requested = args.smiles
    elif args.smiles_file is not None:
        requested = []
        for line in args.smiles_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi, _, cap = line.partition("\t")
            requested.append(smi.strip())
            if cap.strip():
                captions[smi.strip()] = cap.strip()
    else:
        requested = select_by_shift(args.cohort, available, args.top_shift)

    selected, missing = resolve_smiles(requested, available)
    skipped = [f"{m}: not present in both attribution files" for m in missing]
    if not selected:
        sys.exit("none of the requested molecules were found in both "
                 "attribution files")

    parts = [
        f"<html><head><meta charset='utf-8'><title>{html.escape(args.title)}"
        "</title></head><body style='font:14px system-ui;max-width:1500px;"
        "margin:24px auto;padding:0 16px'>"
        f"<h2>{html.escape(args.title)}</h2>"
        "<p style='color:#555;max-width:900px'>Values are signed per-atom "
        "contributions to the logit difference from an all-padding baseline: "
        "red toward the positive class, green away from it. Panels within a "
        "molecule share a colour scale except the descriptor-pathway panel, "
        "which is scaled separately; each panel prints its own range. An "
        "attribution is a contribution to a sum, not a per-atom "
        "classification.</p>"]

    drawn = 0
    for smiles in selected:
        a = a_df[a_df.smiles == smiles]
        b = b_df[b_df.smiles == smiles]
        folds = sorted(set(a.fold) & set(b.fold))
        if args.fold is not None:
            if args.fold not in folds:
                skipped.append(f"{smiles}: fold {args.fold} not present "
                               f"(available: {folds})")
                continue
            fold = args.fold
        else:
            fold = folds[0]
        a = a[a.fold == fold].sort_values("atom_index")
        b = b[b.fold == fold].sort_values("atom_index")

        # Non-finite attributions cannot be drawn: they propagate into the
        # colour scale and RDKit rejects them. Report and move on.
        n_bad = int(a.attr_total.isna().sum() + b.attr_total.isna().sum()
                    + b.attr_descriptor.isna().sum())
        if n_bad:
            skipped.append(f"{smiles}: {n_bad} non-finite attribution "
                           f"value(s) in fold {fold}")
            continue

        a_tot = a.attr_total.tolist()
        b_tok, b_desc = b.attr_token.tolist(), b.attr_descriptor.tolist()
        b_tot = b.attr_total.tolist()

        label = int(b.label.iloc[0]) if "label" in b else None
        tok_mass = b.attr_token.abs().sum()
        ratio = b.attr_descriptor.abs().sum() / max(tok_mass, 1e-9)

        shared = max(max(abs(v) for v in a_tot), max(abs(v) for v in b_tot))
        dscale = max((abs(v) for v in b_desc), default=0.0) or 1e-9

        meta = [f"fold {fold}"]
        if label is not None:
            meta.append(f"label {label}")
        meta.append(f"descriptor pathway {ratio * 100:.1f}% of token mass")

        parts.append(
            "<div style='border:1px solid #ddd;border-radius:8px;"
            "margin:18px 0;padding:12px'>"
            f"<div style='font:600 15px system-ui'>"
            f"<code style='font-weight:400'>{html.escape(smiles)}</code></div>"
            f"<div style='font:13px system-ui;color:#444;margin:4px 0 8px'>"
            f"{' &nbsp;|&nbsp; '.join(meta)}</div>")
        if smiles in captions:
            parts.append(
                "<div style='font:13px system-ui;color:#666;margin-bottom:8px;"
                f"max-width:1000px'>{html.escape(captions[smiles])}</div>")
        parts.append("<div style='display:flex;flex-wrap:wrap;"
                     "justify-content:center'>")

        parts.append(draw_panel(smiles, a_tot, shared, args.label_a,
                                args.width, args.height))
        parts.append(draw_panel(smiles, b_tot, shared, args.label_b,
                                args.width, args.height))
        if not args.no_pathways:
            parts.append(draw_panel(smiles, b_tok, shared,
                                    "↳ token pathway",
                                    args.width, args.height))
            parts.append(draw_panel(smiles, b_desc, dscale,
                                    "↳ descriptor pathway",
                                    args.width, args.height, decimals=3))
        parts.append("</div>")
        if not args.no_table:
            parts.append(atom_table(smiles, a_tot, b_tok, b_desc, b_tot,
                                    args.label_a, args.label_b))
        parts.append("</div>")
        drawn += 1

    for message in skipped:
        print(f"skipped {message}", file=sys.stderr)
    if skipped and args.strict:
        sys.exit(f"--strict: {len(skipped)} molecule(s) could not be drawn")
    if not drawn:
        sys.exit("no molecules could be drawn")

    parts.append("</body></html>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"{drawn} molecule(s) drawn -> {args.out}"
          + (f" ({len(skipped)} skipped)" if skipped else ""))


if __name__ == "__main__":
    main()
