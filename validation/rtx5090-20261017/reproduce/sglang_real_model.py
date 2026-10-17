"""Real pretrained ModelOpt FP8 cold/cache-hit scoring and generation."""
import argparse
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent

def score_rows(outputs):
    if isinstance(outputs, dict):
        outputs = [outputs]
    rows = [[float(x[0]) for x in o["meta_info"]["output_token_ids_logprobs"][0]] for o in outputs]
    assert all(math.isfinite(v) for row in rows for v in row)
    return rows

def max_delta(a, b):
    return max(abs(x-y) for ar, br in zip(a, b, strict=True) for x, y in zip(ar, br, strict=True))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["baseline", "patched"])
    args = ap.parse_args()
    graph = os.environ.get("PR5090_CUDA_GRAPH") == "1"
    tag = args.arm + ("-graph" if graph else "")
    import sglang as sgl
    from transformers import AutoTokenizer
    model = str(ROOT / "models/Qwen3-8B-FP8")
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    started = time.monotonic()
    engine = sgl.Engine(model_path=model, dtype="bfloat16", tp_size=1,
                        context_length=1024, mem_fraction_static=0.65,
                        max_running_requests=4, chunked_prefill_size=512,
                        attention_backend="triton", enable_deterministic_inference=True,
                        disable_cuda_graph=not graph, cuda_graph_max_bs_decode=4,
                        disable_overlap_schedule=True,
                        log_level="info", random_seed=0)
    loaded = time.monotonic() - started
    try:
        selected = [tok.encode(x, add_special_tokens=False)[0] for x in ["A", "B", "C"]]
        cases = []
        for trial in range(3):
            engine.flush_cache()
            prefix = tok.encode("<|im_start|>system\nSelect the correct option. Answer with one letter only.<|im_end|>\n<|im_start|>user\n" +
                                f"Trial {trial}. An apple was picked from an orchard. " * (trial+1) + "\n", add_special_tokens=False)
            branches = [tok.encode(text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False) for text in [
                "Which item was picked?\nA. An apple\nB. A stone\nC. A river\n",
                "Which item was picked?\nA. A river\nB. A stone\nC. An apple\n",
            ]]
            engine.generate(input_ids=prefix, sampling_params={"temperature": 0, "max_new_tokens": 1})
            inputs = [prefix + b for b in branches]
            runs, cached, timings = [], [], []
            for _ in range(3):
                start = time.perf_counter()
                outputs = engine.generate(input_ids=inputs,
                    sampling_params={"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
                    return_logprob=True, token_ids_logprob=[selected[:] for _ in inputs], logprob_start_len=-1)
                timings.append(time.perf_counter()-start)
                runs.append(score_rows(outputs))
                cached.append([o["meta_info"].get("cached_tokens") for o in outputs])
            row = {"trial": trial, "prefix_tokens": len(prefix), "branch_tokens": [len(x) for x in branches],
                   "runs": runs, "cached_tokens": cached, "seconds": timings,
                   "cold_warm_delta": max_delta(runs[0], runs[1]), "warm_repeat_delta": max_delta(runs[1], runs[2])}
            cases.append(row)
            print("REAL_FP8_CACHE_CASE", json.dumps(row), flush=True)
        texts = ["Calculate 17 * 23. Give only the number.", "用中文用两句话解释为什么天空是蓝色的。",
                 "Write a Python function to check whether a string is a palindrome.", "Explain in one sentence why caching helps inference."]
        prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                   add_generation_prompt=True, enable_thinking=False) for t in texts]
        generations = engine.generate(prompt=prompts, sampling_params={"temperature": 0, "max_new_tokens": 48}, return_logprob=True, top_logprobs_num=5)
        durations = []
        for _ in range(3):
            engine.flush_cache()
            start = time.perf_counter()
            engine.generate(prompt=prompts, sampling_params={"temperature": 0, "max_new_tokens": 24, "ignore_eos": True})
            durations.append(time.perf_counter()-start)
        report = {"arm": args.arm, "model": "nvidia/Qwen3-8B-FP8", "revision": "2cebc4c89e25abc17668c81b01dceaf3d8b914d5",
                  "sglang": sgl.__version__, "mode": "full engine, deterministic inference, Triton attention, TP=1", "cuda_graph": graph,
                  "load_seconds": loaded, "cache_cases": cases, "generation": generations,
                  "repeat_four_prompt_seconds": durations}
        (ROOT / f"sglang-real-{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print("REAL_FP8_FINISHED", args.arm, flush=True)
    finally:
        engine.shutdown()

if __name__ == "__main__":
    main()
