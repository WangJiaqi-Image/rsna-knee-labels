"""End-to-end example: label a train.csv with the rule extractor, then check it against
whatever gold-annotated studies it contains.

Run from the repo root after `pip install -e .`:
    python examples/quickstart.py /path/to/train.csv
"""

import sys

import pandas as pd

from rsna_knee_labels import TARGETS, extract, score_against_gold

if __name__ == "__main__":
    train_df = pd.read_csv(sys.argv[1])

    # Label every study with the rule extractor.
    labels = pd.DataFrame([extract(r) for r in train_df["Report"].fillna("")])
    labels["StudyInstanceUID"] = train_df["StudyInstanceUID"].values
    labels = labels.set_index("StudyInstanceUID")
    print(labels[TARGETS].head())

    # Check those labels against whichever studies in train_df carry full gold columns.
    score_against_gold(train_df)
