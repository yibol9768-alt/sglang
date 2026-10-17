"""Compare a complete paired frozen-question evaluation, retaining disagreements."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("baseline", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
a, b = [json.loads(p.read_text()) for p in [args.baseline, args.candidate]]
assert a["completed"] == a["count"] == b["completed"] == b["count"]
assert a["source_sha256"] == b["source_sha256"]
assert a["max_output_tokens"] == b["max_output_tokens"]
rows = []
for x, y in zip(a["rows"], b["rows"], strict=True):
    assert x["dataset_index"] == y["dataset_index"] and x["question"] == y["question"]
    rows.append({"index": x["dataset_index"], "baseline_correct": x["correct"],
        "candidate_correct": y["correct"], "text_equal": x["text"] == y["text"],
        "baseline_missing": x["predicted"] is None, "candidate_missing": y["predicted"] is None,
        "baseline_finish": x["finish_reason"], "candidate_finish": y["finish_reason"]})
result = {"count": len(rows), "baseline_correct": a["correct"], "candidate_correct": b["correct"],
    "both_correct": sum(r["baseline_correct"] and r["candidate_correct"] for r in rows),
    "baseline_only_correct": sum(r["baseline_correct"] and not r["candidate_correct"] for r in rows),
    "candidate_only_correct": sum(not r["baseline_correct"] and r["candidate_correct"] for r in rows),
    "both_wrong": sum(not r["baseline_correct"] and not r["candidate_correct"] for r in rows),
    "baseline_missing": a["missing_final"], "candidate_missing": b["missing_final"],
    "identical_texts": sum(r["text_equal"] for r in rows), "rows": rows,
    "scope": "One paired regression check, not evidence of general accuracy improvement or statistical equivalence."}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
