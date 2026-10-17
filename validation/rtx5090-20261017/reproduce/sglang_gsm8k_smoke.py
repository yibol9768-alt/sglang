"""A small, auditable GSM8K check; not a full benchmark or quality guarantee."""
import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent

def number(text):
    try:
        return str(Decimal(text.replace(",", "").replace("$", "").strip().rstrip(".")))
    except InvalidOperation:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["baseline", "patched"])
    args = ap.parse_args()
    import sglang as sgl
    from transformers import AutoTokenizer
    dataset = json.loads((ROOT/"gsm8k-first32.json").read_text())
    model = str(ROOT/"models/Qwen3-8B-FP8")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    engine = sgl.Engine(model_path=model, dtype="bfloat16", tp_size=1, context_length=1024,
        mem_fraction_static=.65, max_running_requests=4, chunked_prefill_size=512,
        attention_backend="triton", enable_deterministic_inference=True,
        disable_cuda_graph=True, disable_overlap_schedule=True, random_seed=0, log_level="warning")
    try:
        prompts = [tokenizer.apply_chat_template([{"role":"user", "content":
            "Solve the problem. Keep the reasoning brief. End with 'Answer: <number>'.\n\n"+row["question"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in dataset["rows"]]
        assert all(len(tokenizer.encode(p))+256 <= 1024 for p in prompts)
        start = time.perf_counter()
        outputs = engine.generate(prompt=prompts, sampling_params={"temperature":0, "max_new_tokens":256})
        elapsed = time.perf_counter()-start
        cases = []
        for index, (source, output) in enumerate(zip(dataset["rows"], outputs, strict=True)):
            matches = re.findall(r"Answer:\s*\*{0,2}\s*\$?([-+]?\d[\d,.]*)", output["text"], flags=re.I)
            predicted = number(matches[-1]) if matches else None
            expected = number(source["answer"].split("####")[-1])
            cases.append({"index":index, "question":source["question"], "expected":expected,
                "predicted":predicted, "correct":predicted is not None and Decimal(predicted)==Decimal(expected),
                "generated_text":output["text"], "meta_info":output["meta_info"]})
        report = {"arm":args.arm, "model":"nvidia/Qwen3-8B-FP8", "scope":"first 32 GSM8K test examples; zero-shot, greedy, thinking disabled; 256 output tokens; exact decimal match after explicit Answer: extraction",
            "source":dataset["source"], "dataset_file_sha256":dataset["whole_file_sha256"],
            "correct":sum(c["correct"] for c in cases), "total":len(cases), "seconds":elapsed, "cases":cases}
        (ROOT/f"sglang-gsm8k32-{args.arm}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print("GSM8K_SMOKE", json.dumps({k:v for k,v in report.items() if k!="cases"}), flush=True)
    finally:
        engine.shutdown()

if __name__ == "__main__":
    main()
