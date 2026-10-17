"""Long-prefix cache invariance and repeated fixed-work FP8 model measurements."""
import argparse
import json
import hashlib
import math
import os
from pathlib import Path
import statistics
import signal
import time
from cool_gpu import wait as cool

ROOT = Path(__file__).resolve().parent


def timed_out(signum, frame):
    raise TimeoutError("A model request made no completion within 120 seconds")


def scores(outputs):
    if isinstance(outputs, dict):
        outputs = [outputs]
    return [[float(v[0]) for v in o["meta_info"]["output_token_ids_logprobs"][0]] for o in outputs]


def difference(a, b):
    return max(abs(x - y) for ar, br in zip(a, b, strict=True) for x, y in zip(ar, br, strict=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["baseline", "patched"])
    ap.add_argument("--tag")
    args = ap.parse_args()
    graph = os.environ.get("PR5090_CUDA_GRAPH") == "1"
    tag = args.tag or args.arm + ("-graph" if graph else "-eager")
    import sglang as sgl
    from transformers import AutoTokenizer
    model = str(ROOT / "models/Qwen3-8B-FP8")
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    engine = sgl.Engine(model_path=model, dtype="bfloat16", tp_size=1,
                        context_length=6144, mem_fraction_static=0.65,
                        max_running_requests=4, chunked_prefill_size=8192,
                        attention_backend="triton", enable_deterministic_inference=True,
                        disable_cuda_graph=not graph, cuda_graph_max_bs_decode=4,
                        disable_prefill_cuda_graph=not graph,
                        cuda_graph_bs_prefill=[4, 128, 256, 1024, 2048, 4096, 8192] if graph else None,
                        disable_overlap_schedule=True, log_level="info", random_seed=0)
    rows = []
    perf = []
    signal.signal(signal.SIGALRM, timed_out)
    try:
        start = tok.encode("<|im_start|>system\nRead the record and answer the question.<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
        filler = tok.encode("The record says an apple was picked from an orchard, and the apple was red. ", add_special_tokens=False)
        selected = [tok.encode(x, add_special_tokens=False)[0] for x in ["A", "B", "C", "D"]]
        for prefix_len, batch in [(128, 1), (1024, 4), (4096, 2)]:
            prefix = start + (filler * (prefix_len // len(filler) + 1))[:prefix_len - len(start)]
            branches = [tok.encode(f"Record {i}. Which item was picked? A. An apple B. A stone C. A river D. A car. Answer with one letter.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False) for i in range(batch)]
            inputs = [prefix + b for b in branches]
            for repetition in range(2):
                cool()
                engine.flush_cache()
                runs, cached = [], []
                for _ in range(3):
                    signal.alarm(120)
                    out = engine.generate(input_ids=inputs,
                        sampling_params={"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
                        return_logprob=True, token_ids_logprob=[selected[:] for _ in inputs], logprob_start_len=-1)
                    signal.alarm(0)
                    if isinstance(out, dict):
                        out = [out]
                    runs.append(scores(out))
                    cached.append([o["meta_info"].get("cached_tokens") for o in out])
                row = {"prefix_tokens": prefix_len, "batch": batch, "repetition": repetition,
                       "input_lengths": [len(v) for v in inputs], "runs": runs, "cached_tokens": cached,
                       "cold_warm_max_abs": difference(runs[0], runs[1]),
                       "warm_repeat_max_abs": difference(runs[1], runs[2])}
                assert all(math.isfinite(v) for r in runs for values in r for v in values)
                rows.append(row)
                print("EXTENDED_CACHE", json.dumps(row), flush=True)
        for prompt_tokens, batch in [(128, 1), (128, 4), (2048, 1), (2048, 4)]:
            prefix = start + (filler * (prompt_tokens // len(filler) + 1))[:prompt_tokens - len(start)]
            inputs = [prefix + tok.encode(f"Case {i}: Summarize the record.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False) for i in range(batch)]
            durations = []
            for repeat in range(6):
                cool()
                signal.alarm(120)
                engine.flush_cache()
                started = time.perf_counter()
                result = engine.generate(input_ids=inputs, sampling_params={"temperature": 0,
                    "max_new_tokens": 32, "min_new_tokens": 32, "ignore_eos": True})
                elapsed = time.perf_counter() - started
                signal.alarm(0)
                if repeat:
                    durations.append(elapsed)
            row = {"prefix_tokens": prompt_tokens, "batch": batch, "generated_tokens_per_request": 32,
                   "repetitions": 5, "warmups": 1, "seconds": durations,
                   "median_s": statistics.median(durations), "min_s": min(durations), "max_s": max(durations)}
            perf.append(row)
            print("EXTENDED_PERF", json.dumps(row), flush=True)
        report = {"arm": args.arm, "cuda_graph": graph, "model": "nvidia/Qwen3-8B-FP8",
                  "revision": "2cebc4c89e25abc17668c81b01dceaf3d8b914d5", "sglang": sgl.__version__,
                  "mode": "TP1, deterministic, BF16, Triton attention, prefill ceiling 8192; cooldown <=55C before each measured repetition",
                  "prefill_graph_capture_tokens": [4, 128, 256, 1024, 2048, 4096, 8192] if graph else [],
                  "helper_source_sha256": hashlib.sha256((ROOT / "sources/sglang_fp8_utils.py").read_bytes()).hexdigest() if args.arm == "patched" else None,
                  "cache_cases": rows, "fixed_work_performance": perf}
        (ROOT / f"sglang-fp8-extended-{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        signal.alarm(0)
        engine.shutdown()


if __name__ == "__main__":
    main()
