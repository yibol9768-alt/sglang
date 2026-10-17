"""Compare repeated real-model runs without hiding the unmodified variance."""
from pathlib import Path
import argparse
import json
import statistics

DEFAULT=Path(__file__).resolve().parent.parent / "outputs/5090-pr-candidates/evidence"
if not DEFAULT.exists():
    DEFAULT=Path(__file__).resolve().parent.parent / "evidence"
parser=argparse.ArgumentParser()
parser.add_argument("--evidence-dir", type=Path, default=DEFAULT)
ROOT=parser.parse_args().evidence_dir
names=["resident","baseline","patched","patched-graph","baseline-repeat","patched-repeat"]
data={n:json.loads((ROOT/f"nvfp4-real-{n}.json").read_text()) for n in names}

def compare(a,b):
    x,y=data[a],data[b]
    xw={v["name"]:v["sha256"] for v in x["experts"]}
    yw={v["name"]:v["sha256"] for v in y["experts"]}
    first=[]
    shared=[]
    for tx,ty,lx,ly in zip(x["tokens"],y["tokens"],x["logprobs"],y["logprobs"]):
        for token in lx[0].keys()&ly[0].keys():
            first.append(abs(lx[0][token]-ly[0][token]))
        for i,(ax,ay) in enumerate(zip(lx,ly)):
            if tx[:i]!=ty[:i]:break
            shared.extend(abs(ax[t]-ay[t]) for t in ax.keys()&ay.keys())
    result={"arms":[a,b],"same_expert_hashes":xw==yw,"expert_tensors":len(xw),
        "identical_token_sequences":sum(ax==ay for ax,ay in zip(x["tokens"],y["tokens"])),
        "prompts":len(x["tokens"]),"first_step_common_topk_max_abs_logprob_difference":max(first,default=0),
        "same_prefix_common_topk_max_abs_logprob_difference":max(shared,default=0)}
    if "expert_scales" in x and "expert_scales" in y:
        xs={v["name"]:v["sha256"] for v in x["expert_scales"]}
        ys={v["name"]:v["sha256"] for v in y["expert_scales"]}
        result.update({"same_expert_scale_hashes":xs==ys,"scale_parameters":len(xs)})
    return result

summary={
 "scope":"Real pretrained Qwen3-30B-A3B NVFP4; TP1, Marlin MoE/Cutlass dense, eight greedy prompts capped at 48 output tokens. Repeated starts expose unmodified output variation; no bitwise output-equivalence claim.",
 "capacity":{"extra_cuda_allocated_bytes_freed":data["baseline"]["allocated_bytes"]-data["patched"]["allocated_bytes"],
    "restored_expert_bytes":data["patched"]["offloaded_expert_bytes"],
    "extra_freed_equals_restored":data["baseline"]["allocated_bytes"]-data["patched"]["allocated_bytes"]==data["patched"]["offloaded_expert_bytes"]},
 "comparisons":[compare(a,b) for a,b in [("resident","baseline"),("baseline","patched"),("baseline","baseline-repeat"),("patched","patched-repeat"),("baseline-repeat","patched-repeat"),("patched","patched-graph")]],
 "timing_medians_seconds":{n:statistics.median(v["repeat_four_prompt_seconds"]) for n,v in data.items()},
 "gsm8k32":{n:{"correct":data[n]["gsm8k32"]["correct"],"total":data[n]["gsm8k32"]["total"],
    "missing_explicit_answer":sum(r["predicted"] is None for r in data[n]["gsm8k32"]["rows"]),
    "incorrect_indices":[i for i,r in enumerate(data[n]["gsm8k32"]["rows"]) if not r["correct"]]}
    for n in ["baseline-repeat","patched-repeat"]},
 "limits":"The 32 examples are a small smoke check, not a full accuracy benchmark or proof of numerical equivalence. The same dataset, prompt, extraction and token cap are used in both repeated arms. Graph is candidate compatibility coverage, not a same-mode baseline comparison.",
}
assert all(c["same_expert_hashes"] for c in summary["comparisons"])
assert summary["capacity"]["extra_freed_equals_restored"]
(ROOT/"nvfp4-real-comparison.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
