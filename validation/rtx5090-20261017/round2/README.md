# RTX 5090: final second-round validation

This evidence supports SGLang #44861 and vLLM #62146. It contains no model weights, credentials or email data. Hardware was one RTX 5090, 400 W, WSL2; the guard terminates only the test process group at 72°C. Final successful runs stayed below that threshold. Cooling is outside measured intervals.

## Final results

| Check | Result |
|---|---|
| SGLang complete existing SM120 test file | 137 passed |
| SGLang prefixes 128/1024/4096, eager and Graph, repeated cold/cache comparisons | All candidate score differences zero |
| M-only row tiling versus initial fixed32 candidate | 120 kernel comparisons exact; all 128 complete evaluation texts identical |
| SGLang independent GSM8K rows32–159, cap512 | Baseline119/128, candidate117/128; missing numeric answers1/2 |
| Exact-base vLLM registered parameter hashes | 1299/1299 match |
| Exact-base vLLM additional restored expert bytes | 1,107,296,256 =1.03125GiB |
| Exact-base vLLM CUDA allocation after initialization | 19,267,828,224 →18,160,531,968 bytes; same delta |
| Exact-base vLLM GSM8K rows32–159, cap512 | Baseline122/128, candidate121/128; missing numeric answers0/0 |
| Exact-base vLLM canonical-order control | 20 raw-logit comparisons exact; 12 generated requests have identical tokens and top5 logprobs across UVA/GPU/restored-UVA |
| Exact-base native residency regression | Baseline2fail, candidate2pass, normal repository fixtures |
| Exact-base Graph fixed4×24-token benchmark, five measured repetitions | 0.263s →1.128s median, approximately4.29× wall time |

These are bounded regression checks, not general accuracy improvements or statistical equivalence. SGLang retains a measured Graph cost of about5–20% across the four tested prefix/batch combinations; see full samples in `evidence/sglang-fp8-extended-*.json`. vLLM exposes a substantial PCIe host-transfer cost once offload actually takes effect. Its default no-offload execution adds no per-token work in this production change.

## Numerical diagnosis and compatibility

Stock repeated vLLM requests vary even before the patch. Layer tracing and real expert replay isolate variation associated with Marlin's routed-token packing order. Freezing only that order removes the measured GEMM/expert differences across residency changes. The final complete-model diagnostic repeats this control on matching official binaries. `stable_marlin_alignment.py` adds host synchronization and is **diagnostic only**: it is excluded from production and performance measurements. Stock inference is not guaranteed bit-exact.

Full non-UVA NVFP4/Marlin inference fails in both exact-base arms with `b_scales is not on GPU`, the cached-scale issue described in open vLLM #43453. The CPU-fallback unit case proves restoration, not full inference support. ROCm, other GPUs/backends and TP>1 were not exercised.

The final vLLM runtime is official `0.32.0rc1.dev323+gce49c47cb` for exact Git base `ce49c47cbb241b59a32ef110079a5261dbd08e49`, PyTorch2.14.0+cu130, FlashInfer0.7.2, Triton3.8.0. Complete PR/base versions of three Python files are installed atomically; no function-definition injection is used. CUDA binaries match the base and the PR changes no CUDA code. SGLang uses0.5.21 compiled dependencies with the changed main definitions, whose affected baseline ASTs match main; it is not a complete main rebuild.

Historical v0.31/hybrid-main attempts are retained with their original limitations. They expose1443 registered parameters versus1299 in the final build; hashes are compared within each runtime pair. Early default large-Graph thermal stops, an isolated8GiB compilation failure, and setup/cache errors are failed attempts, never counted as successful validation.

## Reproduction

Pin model revisions: `nvidia/Qwen3-8B-FP8@2cebc4c89e25abc17668c81b01dceaf3d8b914d5` and `RedHatAI/Qwen3-30B-A3B-NVFP4@a4df9f017835830cbb588cfe72bc4e50294183c0`. The frozen128 questions are disjoint from the previous32; their source hash and MIT license are included.

On a Linux/systemd setup, copy `reproduce/` into a task's `gpu_payload/`, place `selected-vllm/` directly under the task root, and prepare the model directories named by the scripts. Set `PR5090_TASK_ROOT` and adapt runtime/header paths in the wrapper. The wrapper expects400W, uses a shared exclusive lock, an8GiB test scope, and72°C protection; it changes no hardware settings or services. Create the matching vLLM environment with `install_exact_main.py`; its wheel URL is pinned to the exact base. SGLang dependency/source setup is recorded in the first-round bundle and scripts. Run from `gpu_payload/`:

```bash
bash run_round2.sh vllm-main reproduce-base heldout_model_eval.py baseline --runtime vllm --tag reproduce-base
bash run_round2.sh vllm-main reproduce-patch heldout_model_eval.py patched --runtime vllm --tag reproduce-patch
bash run_round2.sh vllm-main reproduce-control nvfp4_investigate.py patched --tag reproduce-control --stable-alignment --skip-hashes
bash run_round2.sh vllm-main reproduce-graph heldout_model_eval.py patched --runtime vllm --limit 8 --graph --benchmark --tag reproduce-graph
bash run_round2.sh vllm-main reproduce-unit run_vllm_regression_pytest.py patched
bash run_round2.sh sglang reproduce-native run_extended_fp8_pytest.py patched --all
PR5090_CUDA_GRAPH=1 bash run_round2.sh sglang reproduce-fp8 sglang_fp8_extended.py patched --tag reproduce-fp8
```

Use the corresponding baseline arm for paired checks. Source snapshots, final patches, complete outputs, timings, thermal logs, version freeze and publication checks are included. Local validation does not replace upstream CI.
