"""LLM-based weak-label extraction via a hosted API (Alibaba Cloud Bailian / DashScope),
for when a larger flagship model than what fits on a local GPU is worth the API cost.

Same prompt and output format as labeling.llm_based (which runs a smaller model locally
via vLLM), so the two are interchangeable -- run one, then the other, and diff the
results with evaluation.score_against_gold to see whether the larger model is actually
worth its cost for this corpus.

Competition rules permit this: hosted LLM inference on report text is explicitly not
"private sharing of Competition Data", provided the service is reasonably accessible to
every participant and low-cost (see the competition's "Use of Commercially Hosted LLMs"
ruling). You are responsible for checking this holds for whatever competition or dataset
you use this on.

Setup:
    pip install "rsna-knee-labels[api]"
    export DASHSCOPE_API_KEY="..."   # never hardcode this or paste it into a prompt/chat

CLI usage:
    # cheap sanity check first: just the gold-annotated rows
    python -m rsna_knee_labels.labeling.llm_api_based --train-csv train.csv --studies gold
    python -m rsna_knee_labels.labeling.llm_api_based --train-csv train.csv --studies all \
        --model qwen-max -o label_table_qwen_max.csv

Library usage:
    from rsna_knee_labels.labeling.llm_api_based import run
    label_table = run(studies_df, model="qwen-max")
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from .llm_based import SYSTEM_PROMPT, parse_response
from .rule_based import TARGETS

# DashScope's OpenAI-compatible endpoint. Model ids as of writing: "qwen-max" (flagship,
# closed-weight) and open-weight MoE options such as "qwen3-235b-a22b" (cheaper per call,
# fewer active parameters). Check the Bailian console for the current model list/pricing
# before committing to a full-corpus run.
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-max"


def _client():
    from openai import OpenAI

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Export it in your shell -- never hardcode an "
            "API key in code or paste it into a chat/prompt.")
    return OpenAI(api_key=key, base_url=BASE_URL)


def label_one(client, report: str, model: str = DEFAULT_MODEL, retries: int = 3) -> dict:
    """One API call, with the same parse_response fallback llm_based.py uses on a
    malformed reply, plus a small retry loop for transient API errors."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Report:\n{report or '(empty report)'}"},
    ]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=800)
            return parse_response(resp.choices[0].message.content or "")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def run(studies_df: pd.DataFrame, model: str = DEFAULT_MODEL,
       sleep_between: float = 0.0) -> pd.DataFrame:
    """Label every row of studies_df (needs StudyInstanceUID + Report columns).

    Sequential, not batched -- the API bills and rate-limits per call regardless, and a
    thread pool is easy to add later if throughput becomes the bottleneck rather than
    cost. `sleep_between` is a courtesy delay if you hit rate limits.
    """
    client = _client()
    rows = []
    for i, (uid, report) in enumerate(
            zip(studies_df["StudyInstanceUID"], studies_df["Report"].fillna(""))):
        row = label_one(client, report, model=model)
        row["StudyInstanceUID"] = uid
        rows.append(row)
        if sleep_between:
            time.sleep(sleep_between)
        if (i + 1) % 20 == 0 or i + 1 == len(studies_df):
            print(f"[llm_api_based] {i + 1}/{len(studies_df)} done", flush=True)
    return pd.DataFrame(rows).set_index("StudyInstanceUID")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True,
                   help="CSV with StudyInstanceUID, Report, and (for --studies gold) the "
                        "12 gold target columns from rule_based.TARGETS")
    p.add_argument("--studies", choices=["gold", "all"], default="gold",
                   help="'gold' labels only rows with a complete set of gold columns -- "
                        "run this first, it costs almost nothing, before spending API "
                        "budget on the full corpus")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="DashScope model id, e.g. qwen-max or qwen3-235b-a22b")
    p.add_argument("-o", "--output", default=None,
                   help="output CSV path; defaults to label_table_<model>_<studies>.csv")
    p.add_argument("--sleep-between", type=float, default=0.0,
                   help="seconds to sleep between calls, if you hit rate limits")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    if args.studies == "gold":
        df = train_df[train_df[TARGETS].notna().all(axis=1)]
    else:
        df = train_df
    print(f"[llm_api_based] labeling {len(df)} report(s) with {args.model}")
    result = run(df, model=args.model, sleep_between=args.sleep_between)
    out_path = args.output or f"label_table_{args.model.replace('/', '_')}_{args.studies}.csv"
    result.to_csv(out_path)
    print(f"[llm_api_based] wrote {out_path}")
