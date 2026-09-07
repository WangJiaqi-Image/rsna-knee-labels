"""LLM-based weak-label extraction from radiology reports, using a local open-weight model.

Alternative to the regex lexicon in rule_based.py. Rather than enumerating vocabulary per
language, a general-purpose multilingual instruction model reads each report and judges
clinical significance directly -- several reports name a finding explicitly ("mild
synovitis") while the gold annotation still reads negative, because the annotator's
threshold is "clinically notable", not "mentioned at all". A regex either fires on the
word or it doesn't; a model can be told to judge significance the way the annotator did.

Runs entirely locally via vLLM -- report text is not sent to any external API, which
matters if your data-handling rules (e.g. a competition's) do not allow that.

Install the extra dependencies with: pip install "rsna-knee-labels[llm]"

CLI usage:
    python -m rsna_knee_labels.labeling.llm_based --train-csv train.csv --studies gold
    python -m rsna_knee_labels.labeling.llm_based --train-csv train.csv --studies all -o out.csv

Library usage:
    from rsna_knee_labels.labeling.llm_based import run
    label_table = run(studies_df)   # studies_df needs StudyInstanceUID + Report columns
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from .rule_based import TARGETS

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

FINDING_DEFS = {
    "ACL": "anterior cruciate ligament tear, sprain or other injury",
    "MCL": "medial (tibial) collateral ligament tear, sprain or other injury",
    "Medial Meniscus": "tear or other clinically relevant injury of the medial meniscus",
    "Lateral Meniscus": "tear or other clinically relevant injury of the lateral meniscus",
    "Medial OA": "osteoarthritis (cartilage loss, chondromalacia, osteophytes, joint space "
                 "narrowing) in the MEDIAL femorotibial compartment",
    "Lateral OA": "osteoarthritis in the LATERAL femorotibial compartment",
    "PF OA": "osteoarthritis in the patellofemoral compartment",
    "Effusion": "joint effusion (fluid within the knee joint)",
    "Synovitis": "synovitis / synovial thickening or proliferation that is clinically "
                 "notable -- not an incidental trace signal mentioned only in passing",
    "Baker's": "Baker's (popliteal) cyst",
    "Contusion": "acute traumatic bone contusion / bone marrow oedema -- NOT reactive "
                 "subchondral oedema beneath a degenerative/osteoarthritic cartilage defect",
    "Fracture": "a fracture -- NOT a 'microfracture' surgical procedure and NOT a "
                "statement about future fracture risk",
}

SYSTEM_PROMPT = f"""You are assisting a radiologist by structuring the findings already \
stated in a knee MRI report. The report may be written in any language (English, \
Spanish, French, Dutch, German, Turkish, Croatian/Serbian/Bosnian, Greek, or \
Bulgarian/Russian).

For each of the twelve findings below, read the report and judge whether a radiologist \
formally grading this study would mark it present. A finding mentioned only as an \
incidental, trace, or otherwise clinically insignificant observation should usually be \
graded as NOT present, even if the word appears in the text -- annotators grade the \
overall clinical picture, not literal keyword presence. A finding not mentioned at all \
should get a score near, but not exactly, 0 (the report may simply be silent on it).

Findings:
{chr(10).join(f"- {k}: {v}" for k, v in FINDING_DEFS.items())}

For each finding, output:
- "score": your probability (0.0-1.0) that a radiologist would grade this finding present
- "confidence": your confidence (0.0-1.0) in that judgment, given how explicit the report is

Respond with ONLY a JSON object, no other text, in exactly this shape (all twelve finding
names, spelled exactly as given above):
{{"ACL": {{"score": 0.0, "confidence": 0.0}}, "MCL": {{"score": 0.0, "confidence": 0.0}}, ...}}
"""

_JSON_RX = re.compile(r"\{.*\}", re.DOTALL)


def build_prompts(reports: list[str], tokenizer) -> list[str]:
    prompts = []
    for report in reports:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Report:\n{report or '(empty report)'}"},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))
    return prompts


def parse_response(text: str) -> dict:
    """Extract the JSON object from a model response; returns baseline values on failure."""
    m = _JSON_RX.search(text)
    out = {}
    parsed = {}
    if m:
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            parsed = {}
    for t in TARGETS:
        entry = parsed.get(t, {}) if isinstance(parsed, dict) else {}
        score = entry.get("score") if isinstance(entry, dict) else None
        conf = entry.get("confidence") if isinstance(entry, dict) else None
        out[t] = float(score) if isinstance(score, (int, float)) else 0.28
        out[t + "__conf"] = float(conf) if isinstance(conf, (int, float)) else 0.05
    return out


def run(studies_df: pd.DataFrame, tensor_parallel_size: int = 2,
       batch_size: int = 64) -> pd.DataFrame:
    """Label every row of studies_df (needs StudyInstanceUID + Report columns)."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(model=MODEL_ID, tensor_parallel_size=tensor_parallel_size,
             gpu_memory_utilization=0.85, max_model_len=4096, dtype="bfloat16",
             enforce_eager=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=800)

    reports = studies_df["Report"].fillna("").tolist()
    prompts = build_prompts(reports, tokenizer)

    rows = []
    for i in range(0, len(prompts), batch_size):
        outs = llm.generate(prompts[i:i + batch_size], sampling)
        for uid, out in zip(studies_df["StudyInstanceUID"].iloc[i:i + batch_size], outs):
            row = parse_response(out.outputs[0].text)
            row["StudyInstanceUID"] = uid
            rows.append(row)
        print(f"[llm_based] {min(i + batch_size, len(prompts))}/{len(prompts)} done",
              flush=True)
    return pd.DataFrame(rows).set_index("StudyInstanceUID")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True,
                   help="CSV with StudyInstanceUID, Report, and (for --studies gold) the "
                        "12 gold target columns from rule_based.TARGETS")
    p.add_argument("--studies", choices=["gold", "all"], default="gold",
                   help="'gold' labels only rows with a complete set of gold columns "
                        "(useful for validating this extractor against them); 'all' "
                        "labels every row")
    p.add_argument("-o", "--output", default=None,
                   help="output CSV path; defaults to label_table_llm_<studies>.csv")
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    if args.studies == "gold":
        df = train_df[train_df[TARGETS].notna().all(axis=1)]
    else:
        df = train_df
    print(f"[llm_based] running {MODEL_ID} over {len(df)} report(s)")
    result = run(df, tensor_parallel_size=args.tensor_parallel_size,
                batch_size=args.batch_size)
    out_path = args.output or f"label_table_llm_{args.studies}.csv"
    result.to_csv(out_path)
    print(f"[llm_based] wrote {out_path}")
