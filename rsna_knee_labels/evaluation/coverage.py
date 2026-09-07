"""How often the rule lexicon finds nothing at all, across the whole corpus.

`gold_check.score_against_gold` measures agreement -- is a fired rule *right* -- but only
on the small gold-annotated subset, which rarely has enough sample to arbitrate a lexicon
change. Silence rate measures a different thing -- does a rule *fire at all* -- and needs
no gold labels, so it runs over every report. The two answer different questions:

|              | measures            | sample       | can decide                              |
|--------------|---------------------|--------------|------------------------------------------|
| agreement    | is a fired rule right | small (gold) | whether a target's labels are usable at all |
| silence rate | does a rule fire     | whole corpus | which language/target to work on next   |

A target silent on nearly every report may simply be rare, and silence there is correct.
A target silent in one language and not another is very likely a vocabulary gap in that
language, not a difference in the underlying rate of the finding.

This is specific to the rule lexicon (rsna_knee_labels.labeling.rule_based): an LLM-based
label source has no equivalent notion of "no rule fired" -- it always returns some score
and confidence -- so there is nothing analogous to compute for labeling.llm_based output.

CLI usage:
    python -m rsna_knee_labels.evaluation.coverage --train-csv train.csv
    python -m rsna_knee_labels.evaluation.coverage --train-csv train.csv --by-language
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..labeling.rule_based import TARGETS, extract, normalize


def silence_rate(train_df: pd.DataFrame) -> pd.Series:
    """Fraction of all reports where the rule lexicon asserted neither presence nor
    absence for each target -- i.e. no rule fired in either direction."""
    rows = pd.DataFrame([extract(r) for r in train_df["Report"].fillna("")])
    return pd.Series({
        t: float(((rows[t + "__npos"] == 0) & (rows[t + "__nneg"] == 0)).mean())
        for t in TARGETS
    })


_LANG_STOPWORDS = {
    "en": ("the", "and", "is", "of", "no", "with"),
    "es": ("el", "la", "de", "sin", "con", "y"),
    "fr": ("le", "la", "de", "et", "sans", "avec"),
    "nl": ("de", "het", "en", "van", "geen", "zonder"),
    "de": ("der", "die", "das", "und", "ohne", "keine"),
    "tr": ("ve", "bir", "bulunmamaktadir", "izlenmemistir", "yok"),
}


def _guess_language(text: str) -> str:
    """Coarse language guess by stopword count, with a margin so an ambiguous report
    stays 'unknown' rather than being assigned to whichever test ran first -- 'la' is as
    common in Spanish as in French, so a plain substring cascade cannot tell them apart.
    Greek and Cyrillic are not covered here (their script alone identifies them)."""
    norm = normalize(text)
    counts = {lang: sum(norm.count(w) for w in words)
             for lang, words in _LANG_STOPWORDS.items()}
    if not any(counts.values()):
        return "unknown"
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[0][1] <= ranked[1][1]:
        return "unknown"
    return ranked[0][0]


def silence_rate_by_language(train_df: pd.DataFrame) -> pd.DataFrame:
    """silence_rate(), broken out per (rough) language, to find where a target's
    vocabulary gap actually is rather than only that it has one."""
    langs = train_df["Report"].fillna("").map(_guess_language)
    out = {}
    for lang in sorted(langs.unique()):
        sub = train_df[langs == lang]
        out[f"{lang} (n={len(sub)})"] = silence_rate(sub)
    return pd.DataFrame(out).T


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True,
                   help="CSV with a Report column (StudyInstanceUID not required here)")
    p.add_argument("--by-language", action="store_true",
                   help="break the silence rate out by a coarse language guess")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    if args.by_language:
        table = silence_rate_by_language(train_df)
        print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    else:
        rate = silence_rate(train_df).sort_values(ascending=False)
        print(rate.to_string(float_format=lambda x: f"{x:.3f}"))
