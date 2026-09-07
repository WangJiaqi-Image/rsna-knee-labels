"""Validate a weak-label source (rule-based or LLM-derived) against gold-annotated studies.

A "gold" study here is any row of train_df whose TARGETS columns are all non-null --
this is how the RSNA knee competition marks the subset it fully hand-annotated, with the
rest carrying only report text. Reports per-target AUC and score distributions, not just
the macro average, because coverage is known to be uneven across languages/finding types
and a single macro number hides which targets are actually being read correctly.

CLI usage:
    python -m rsna_knee_labels.evaluation.gold_check score --train-csv train.csv
    python -m rsna_knee_labels.evaluation.gold_check score --train-csv train.csv --label-table t.csv
    python -m rsna_knee_labels.evaluation.gold_check diagnose --train-csv train.csv ACL
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

from ..labeling.rule_based import TARGETS, extract


def _label_source(train_df: pd.DataFrame, label_table_path: str | None) -> tuple[pd.DataFrame, str]:
    """Score/confidence table for every study: the rule lexicon, with an optional
    label_table CSV's columns overlaid where it has coverage."""
    lex = pd.DataFrame([extract(r) for r in train_df["Report"].fillna("")])
    lex["StudyInstanceUID"] = train_df["StudyInstanceUID"].values
    lex = lex.set_index("StudyInstanceUID")

    source = "rule lexicon (rsna_knee_labels.labeling.rule_based)"
    if label_table_path:
        label_cols = TARGETS + [t + "__conf" for t in TARGETS]
        tab = pd.read_csv(label_table_path).set_index("StudyInstanceUID")
        missing = [c for c in label_cols if c not in tab.columns]
        if missing:
            raise ValueError(f"{label_table_path} is missing columns: {missing[:3]}...")
        hit = lex.index.intersection(tab.index)
        lex.loc[hit, label_cols] = tab.loc[hit, label_cols].values
        source = label_table_path
    return lex, source


def score_against_gold(train_df: pd.DataFrame, label_table_path: str | None = None) -> pd.DataFrame:
    """Per-target AUC, positive rate and mean confidence of one label source vs. gold.

    AUC on the raw graded score (not thresholded) against the binary gold label -- this is
    what a model trained on the weak labels would actually be supervised against, so it
    measures the same thing training will lean on.
    """
    gold = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]
    if gold.empty:
        raise ValueError("no rows in train_df carry a full set of gold TARGETS columns")

    lex, source = _label_source(train_df, label_table_path)

    hit = gold.index.intersection(lex.index)
    missing_gold = gold.index.difference(lex.index)
    if len(missing_gold):
        print(f"[gold_check] {len(missing_gold)} gold studies have no label-source row at "
              f"all (no report?) and are excluded: {list(missing_gold)[:5]}")

    rows = []
    for t in TARGETS:
        y = gold.loc[hit, t].astype(int).values
        s = lex.loc[hit, t].values
        conf = lex.loc[hit, t + "__conf"].values
        auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
        rows.append({
            "target": t, "n": len(hit), "n_pos": int(y.sum()), "auc": auc,
            "mean_score_pos": float(s[y == 1].mean()) if y.sum() else float("nan"),
            "mean_score_neg": float(s[y == 0].mean()) if (y == 0).sum() else float("nan"),
            "mean_conf": float(conf.mean()),
        })
    report = pd.DataFrame(rows).set_index("target")
    print(f"[gold_check] source: {source}")
    print(f"[gold_check] {len(hit)} of {len(gold)} gold studies scored")
    print(report.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"[gold_check] macro AUC vs. gold: {report['auc'].mean():.4f}")
    return report


def worst_misses(train_df: pd.DataFrame, target: str, label_table_path: str | None = None,
                 n: int = 8) -> None:
    """Print the studies a label source got most wrong on one target, report text included.

    For one target: gold-positive studies the source scored lowest (likely missed
    vocabulary, or an LLM judgment call worth reviewing), and gold-negative studies it
    scored highest (likely a false-positive cue) -- so a human can read the actual miss
    and fix the extractor, rather than guessing from the aggregate AUC alone.
    """
    gold = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]
    lex, source = _label_source(train_df, label_table_path)

    hit = gold.index.intersection(lex.index)
    df = lex.loc[hit, [target, target + "__conf"]].rename(
        columns={target: "score", target + "__conf": "conf"})
    df["report"] = train_df.set_index("StudyInstanceUID").loc[hit, "Report"]
    df["gold"] = gold.loc[hit, target].astype(int)

    fn = df[df["gold"] == 1].sort_values("score").head(n)
    fp = df[df["gold"] == 0].sort_values("score", ascending=False).head(n)

    print(f"[gold_check] source: {source}")
    print(f"\n=== {target}: gold POSITIVE, source scored LOWEST (missed evidence?) ===")
    for st, row in fn.iterrows():
        print(f"\n[{st}] score={row['score']:.3f} conf={row['conf']:.3f}")
        print(f"  {row['report']}")

    print(f"\n=== {target}: gold NEGATIVE, source scored HIGHEST (false-positive cue?) ===")
    for st, row in fp.iterrows():
        print(f"\n[{st}] score={row['score']:.3f} conf={row['conf']:.3f}")
        print(f"  {row['report']}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["score", "diagnose"])
    p.add_argument("target", nargs="?", choices=TARGETS,
                   help="required for `diagnose`; ignored for `score`")
    p.add_argument("--train-csv", required=True,
                   help="CSV with StudyInstanceUID, Report, and the 12 gold TARGETS "
                        "columns (null where a study is not gold-annotated)")
    p.add_argument("--label-table", type=str, default=None,
                   help="a labeling.llm_based (or other) output CSV to check instead of "
                        "the rule lexicon")
    p.add_argument("-n", type=int, default=8, help="diagnose: how many of each to show")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    if args.command == "score":
        score_against_gold(train_df, args.label_table)
    else:
        if args.target is None:
            raise SystemExit("`diagnose` needs a target, e.g.: ... diagnose ACL --train-csv ...")
        worst_misses(train_df, args.target, args.label_table, args.n)
