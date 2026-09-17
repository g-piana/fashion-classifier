"""
prepare_tiers.py
================
Turns your CORRECTED CLAUDE VLM labels into training CSVs, and (optionally)
uses a FashionCLIP zero-shot audit only to advise which attributes could be
served zero-shot instead of trained.

IMPORTANT — two data sources, two different jobs:
  * Claude VLM corrected CSV  = the TRAINING LABEL SOURCE (reliable).
                                This is what train.py learns from.
  * audit_zeroshot CSV        = ADVISORY ONLY. Tells you whether FashionCLIP
                                can separate an attribute zero-shot. It is a
                                poor labeler (it under-fires on many
                                embellishments), so it is NEVER used to build
                                training labels — only to flag "this one might
                                not need training".

Input format (Claude VLM corrected CSV)
---------------------------------------
Wide format, one column per attribute:
    name, <cat_attr1>, <cat_attr2>, <bool_attr1>, <bool_attr2>, ..., note
  - Boolean columns hold True/False (embellishments + boolean model details).
  - Categorical columns (e.g. toe_shape, heel_type) hold one class string.

What it writes (per group)
--------------------------
    labels_<category>_<group>.csv      training CSV for train.py
                                       (name, attributes) underscore-joined
    classes_<category>_<group>.yaml    paste-ready classes: list
    counts_<category>.txt              per-attribute positive counts + advice

Trainability rule (from REAL Claude counts, not zero-shot):
    n_pos >= min_positives   -> TRAIN  (goes in classes list)
    n_pos <  min_positives   -> defer  (too few to train yet; serve zero-shot
                                        or wait for failure mining)
If an audit CSV is given, attributes CLIP separates well AND that Claude shows
at a matching rate are flagged "zero-shot candidate" so you may skip training.

Usage
-----
    python src/prepare_tiers.py `
        --vlm-csv   "E:/fashion-data/csv/vlm_shoes_boot_corrected.csv" `
        --category  shoes `
        --embellishments bead bow buckle button chain charme crystal embroidery `
                         eyelet feather flower fringe fur gemstone hardware lace `
                         logo mesh-insert mirror patch pearl pom-pom ribbon rope `
                         sequin stud tassel web `
        --categoricals toe_shape heel_type heel_height `
        --min-positives 8 `
        --audit-csv "E:/fashion-data/csv/audit_zeroshot_shoes_300.csv" `
        --out-dir   "E:/fashion-data/tiers"

--audit-csv is optional. --categoricals lists single-label columns that should
NOT be folded into the multi-label attributes string; each becomes its own
single-label model and is reported separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def parse_audit_signal(audit_csv: Path) -> dict[str, tuple[float, str]]:
    """Recompute {attr: (pct_positive, signal)} from an audit_zeroshot CSV."""
    df = pd.read_csv(audit_csv)
    n = len(df)
    out: dict[str, tuple[float, str]] = {}
    for col in df.columns:
        if not col.endswith("_score"):
            continue
        attr = col[:-6]
        scores = pd.to_numeric(df[col], errors="coerce").dropna()
        pred_col = f"{attr}_pred"
        pct = (100.0 * pd.to_numeric(df[pred_col], errors="coerce").fillna(0).sum()
               / max(n, 1)) if pred_col in df.columns else 0.0
        mean_sc, std_sc = scores.mean(), scores.std()
        if mean_sc > 0.65 or mean_sc < 0.35:
            sig = "STRONG"
        elif std_sc < 0.04:
            sig = "FLAT"
        elif std_sc < 0.07:
            sig = "WEAK"
        else:
            sig = "OK"
        out[attr] = (round(pct, 1), sig)
    return out


def build_multilabel_csv(vlm: pd.DataFrame, bool_attrs: list[str]) -> pd.DataFrame:
    """One row per image: name, attributes (underscore-joined True columns)."""
    present_mask = {a: as_bool(vlm[a]) for a in bool_attrs if a in vlm.columns}
    rows = []
    for i, r in vlm.iterrows():
        present = [a for a in bool_attrs if a in present_mask and present_mask[a].iloc[i]]
        rows.append({"name": r["name"], "attributes": "_".join(present)})
    return pd.DataFrame(rows)


def analyse_group(
    vlm: pd.DataFrame,
    attrs: list[str],
    min_positives: int,
    audit: dict[str, tuple[float, str]] | None,
) -> pd.DataFrame:
    n = len(vlm)
    rows = []
    for a in attrs:
        if a not in vlm.columns:
            continue
        n_pos = int(as_bool(vlm[a]).sum())
        pct = 100.0 * n_pos / max(n, 1)
        trainable = n_pos >= min_positives

        zs_pct, zs_sig = (audit or {}).get(a, (None, None))
        zeroshot_candidate = (
            audit is not None and zs_sig == "STRONG"
            and zs_pct is not None and pct > 0 and zs_pct >= 0.6 * pct
        )

        if not trainable:
            decision = "defer (too few positives)"
        elif zeroshot_candidate:
            decision = "TRAIN (or zero-shot — CLIP tracks it)"
        else:
            decision = "TRAIN"

        rows.append({
            "attr": a, "n_pos": n_pos, "pct": round(pct, 1),
            "trainable": trainable, "zs_pct": zs_pct, "zs_signal": zs_sig,
            "decision": decision,
        })
    return pd.DataFrame(rows).sort_values("n_pos", ascending=False).reset_index(drop=True)


def write_group_outputs(
    analysis: pd.DataFrame,
    training_csv: pd.DataFrame,
    category: str,
    group: str,
    out_dir: Path,
    report: list[str],
) -> None:
    trainable = analysis[analysis["trainable"]]["attr"].tolist()

    cls_path = out_dir / f"classes_{category}_{group}.yaml"
    with open(cls_path, "w", encoding="utf-8") as f:
        f.write(f"# Paste into conf/category/{group}.yaml — trained classes only.\n")
        f.write("classes:\n")
        for a in trainable:
            f.write(f'  - "{a}"\n')

    csv_path = out_dir / f"labels_{category}_{group}.csv"
    training_csv.to_csv(csv_path, index=False)

    report.append(f"\n=== {group.upper()} ({len(analysis)} attributes) ===")
    report.append(f"{'attr':<14} {'Claude n':>8} {'Claude%':>8}  {'ZS%':>6} {'ZSsig':>6}  decision")
    report.append("-" * 74)
    for _, r in analysis.iterrows():
        zs = f"{r['zs_pct']:>5.1f}%" if r["zs_pct"] is not None else "    —"
        zss = r["zs_signal"] or "—"
        report.append(f"{r['attr']:<14} {r['n_pos']:>8} {r['pct']:>7.1f}%  {zs} {zss:>6}  {r['decision']}")
    report.append(f"\n  -> classes file : {cls_path.name}  ({len(trainable)} trained)")
    report.append(f"  -> training CSV : {csv_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build training CSVs from corrected Claude VLM labels; advise tiers from an audit."
    )
    p.add_argument("--vlm-csv", type=Path, required=True,
                   help="Corrected Claude VLM labels (wide, one column per attribute)")
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--embellishments", nargs="+", required=True,
                   help="Embellishment attribute column names")
    p.add_argument("--categoricals", nargs="*", default=[],
                   help="Single-label columns (e.g. toe_shape heel_type) reported separately")
    p.add_argument("--model-details", nargs="*", default=None,
                   help="Boolean model-detail columns. If given, ONLY these form the "
                        "model_details group; other unlisted CSV columns are dropped with "
                        "a note. If omitted, all non-embellishment/categorical columns are used "
                        "(which can sweep in stray columns — prefer naming them).")
    p.add_argument("--min-positives", type=int, default=8,
                   help="Minimum Claude positives to mark an attribute trainable (default: 8)")
    p.add_argument("--audit-csv", type=Path, default=None,
                   help="Optional audit_zeroshot CSV — advisory zero-shot signal only")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vlm = pd.read_csv(args.vlm_csv)
    audit = parse_audit_signal(args.audit_csv) if args.audit_csv else None

    non_attr = {"name", "note"} | set(args.categoricals)
    emb_set = set(args.embellishments)
    if args.model_details is not None:
        md_bool = [c for c in args.model_details if c in vlm.columns]
        claimed = non_attr | emb_set | set(md_bool)
        dropped = [c for c in vlm.columns if c not in claimed]
    else:
        md_bool = [c for c in vlm.columns if c not in non_attr and c not in emb_set]
        dropped = []

    report = [
        "=" * 74,
        f"TRAINING-DATA PREP — {args.category}",
        f"Label source : {args.vlm_csv.name}  (Claude VLM, corrected)  [{len(vlm)} images]",
        f"Trainable if Claude positives >= {args.min_positives}",
        f"Audit advice : {'on — ' + args.audit_csv.name if audit else 'off'}",
        "=" * 74,
    ]
    if dropped:
        report.append(f"\nDROPPED (in CSV but not in any group — e.g. cut attributes): "
                      f"{', '.join(dropped)}")

    # Embellishments (multi-label)
    emb_present = [a for a in args.embellishments if a in vlm.columns]
    emb_analysis = analyse_group(vlm, emb_present, args.min_positives, audit)
    emb_csv = build_multilabel_csv(vlm, emb_present)
    write_group_outputs(emb_analysis, emb_csv, args.category, "embellishments", args.out_dir, report)

    # Boolean model-details (multi-label)
    if md_bool:
        md_analysis = analyse_group(vlm, md_bool, args.min_positives, audit)
        md_csv = build_multilabel_csv(vlm, md_bool)
        write_group_outputs(md_analysis, md_csv, args.category, "model_details", args.out_dir, report)

    # Categorical model-details (one single-label model each)
    if args.categoricals:
        report.append("\n=== CATEGORICAL MODEL-DETAILS (single-label models) ===")
        for c in args.categoricals:
            if c not in vlm.columns:
                continue
            vc = vlm[c].value_counts().to_dict()
            report.append(f"\n  {c}:")
            for k, v in vc.items():
                mark = "" if v >= args.min_positives else "   (few — may merge/defer)"
                report.append(f"    {str(k):<18} {v:>4}{mark}")
            sub = vlm[["name", c]].rename(columns={c: "attributes"})
            sub.to_csv(args.out_dir / f"labels_{args.category}_{c}.csv", index=False)
            report.append(f"    -> training CSV : labels_{args.category}_{c}.csv")

    text = "\n".join(report)
    print(text)
    (args.out_dir / f"counts_{args.category}.txt").write_text(text, encoding="utf-8")
    print(f"\nReport saved -> {args.out_dir / f'counts_{args.category}.txt'}")
    print("\nNEXT:")
    print("  1. Paste each classes_*.yaml into its conf/category file")
    print("  2. preprocess.py + train.py per group and per categorical column")


if __name__ == "__main__":
    main()
