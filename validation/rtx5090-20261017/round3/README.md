# Single-RTX-5090 validation refresh, 2026-10-21

Scope: SGLang #44860 and #44861, and vLLM #62146. Production PR commits are unchanged. One RTX 5090, TP=1, WSL2, 400 W; the guard stops only the test process group at 72 C. Successful runs peaked at 67 C. No upstream CI or maintainer approval is claimed.

## Complete official SGLang nightly

The runtime is the official `0.5.23.dev20261020+g55b2aef7ad` CPython 3.12 wheel, PyTorch 2.14.1, Triton 3.8.0, FlashInfer 0.7.2, and sglang-kernel 0.4.10, including the wheel's Rust extensions. The two PRs' unchanged production diff applies cleanly to the nightly's complete Python files. The native package is imported normally; no function/AST injection or old SGLang compiled dependencies are used in these nightly tests.

The original PR base and this nightly have unrelated source differences. `reproduce/nightly-source-provenance.json` and the complete `nightly-sources/` snapshots record both arms. Both independent patches are selected together by the harness; the FP8 model does not use the diffusion host-budget module, and the host-budget unit tests do not exercise FP8 GEMM.

| Check | Baseline | Candidate |
|---|---|---|
| Cached-suffix tensor comparisons in the existing test file | 4 failed / 8 passed | All 12 pass within the full suite |
| Complete PR SM120 test file, including Graph, strided input, bias and tails | Not rerun in full | **137 passed** |
| Existing host-budget unit file, normal nightly imports | 4 failed / 32 passed | **36 passed** |
| Qwen3-8B-FP8, prefixes 128/1024/4096, batches 1/4/2, two repetitions, eager and Graph | Selected-token cold/warm logprob deltas up to 2.874847412 | **Zero** in all 12 paired cache cases; warm repeats also zero |
| Complete generated answers, GSM8K test rows 32–47, cap 512 | 15/16, no missing final answers | 16/16, no missing final answers |

The 16-question check reuses a subset of round2, not a new independent evaluation. Only 2/16 complete texts match between baseline and candidate. Neither general accuracy improvement nor statistical equivalence is claimed. The earlier 128-question results remain historical mixed-runtime evidence in round2, with their original limits.

The real model uses `nvidia/Qwen3-8B-FP8@2cebc4c89e25abc17668c81b01dceaf3d8b914d5`, BF16 activations and the nightly's automatic FP8 KV cache, deterministic mode, Triton attention, and the normal Rust unified radix cache. Graph logs show actual replay; unsupported partial prefills may fall back to eager.

### Fixed-work timing

32 forced output tokens per request, one warmup, five measured repetitions per cell, cache flushed before each measured request. Passive cooling to at most 55 C is outside the measured interval. Each arm has one engine boot; this is a bounded cost check, not a general speedup claim.

| Prefix / batch | Eager baseline → candidate | Graph baseline → candidate |
|---|---|---|
| 128 / 1 | 1.4381 s → 1.4459 s | 0.2598 s → 0.3088 s |
| 128 / 4 | 2.0762 s → 1.4880 s | 0.3314 s → 0.3503 s |
| 2048 / 1 | 1.4603 s → 1.5094 s | 0.4512 s → 0.5675 s |
| 2048 / 4 | 2.4906 s → 2.0259 s | 0.7812 s → 0.9135 s |

Graph wall time is **5.7–25.8% higher** on this nightly matrix. The round2 Graph cost of about 5–20% belongs to its older runtime. Eager results include faster and slower cells; no universal performance claim follows.

## Host-budget real-model repeat

The same real SANA 600M revision, live nested cgroup, real 512 MiB anonymous sibling, one streamed / 27 resident DiT blocks, and two seeded 512×512 prompts were repeated. The original budget books 41,292,288 persistent pinned bytes; the candidate respects parent headroom and keeps the streamed block pageable. Both generated images are pixel-identical across arms, and both inner and outer cgroups report zero OOM/OOM-kill events.

This SANA repeat retains the prior SGLang 0.5.21 diffusion runtime, a complete main-source budget module, and the same disclosed VAE checkpoint-loading prerequisite in both arms. It is **not** a full-nightly diffusion-model test. The 36 host-budget tests above do use the complete official nightly. No production OOM was deliberately induced; the fixture reclaims unused cache and calibrates the child/parent limits once before the unmodified budget algorithm runs.

## vLLM repeat

The matching official base remains `0.32.0rc1.dev323+gce49c47cb` / `ce49c47cbb241b59a32ef110079a5261dbd08e49`, PyTorch 2.14.0+cu130, FlashInfer 0.7.2. The native residency regression passes both cases.

The real `RedHatAI/Qwen3-30B-A3B-NVFP4@a4df9f017835830cbb588cfe72bc4e50294183c0` completes eight capped generated answers in both arms, scoring 7/8 in each. Restored expert offload increases from zero to **1,107,296,256 bytes**, and CUDA allocation decreases by exactly that amount, from 19,267,828,224 to 18,160,531,968 bytes. This is a subset smoke repeat, not a fresh full evaluation or a new-main rebuild. Stock Marlin packing variation, the non-UVA failure in both arms, and the previously measured 4.29× Graph wall-time tradeoff remain disclosed in round2.

## Reproduction and setup failures

Use the existing round2 model directories, pinned datasets, guard and cooling helpers in `gpu_payload/`. Copy this round's `reproduce/` files and `nightly-sources/` into the isolated task's `round3/`, then copy `round3_fp8_model.py`, `run_full_fp8_pytest.py` and `run_full_host_budget_pytest.py` into `gpu_payload/`. `install_sglang_full.sh` records the official wheel URL and SHA256 and the matching external CUDA 13.0 compiler toolchain. The wheel requires glibc 2.39 and CPython 3.12.

```bash
export PR5090_TASK_ROOT=/absolute/path/to/isolated/pr5090
bash "$PR5090_TASK_ROOT/round3/install_sglang_full.sh"
bash "$PR5090_TASK_ROOT/round3/run_sglang_full.sh" fp8-base baseline run_full_fp8_pytest.py
bash "$PR5090_TASK_ROOT/round3/run_sglang_full.sh" fp8-patch patched run_full_fp8_pytest.py
bash "$PR5090_TASK_ROOT/round3/run_sglang_full.sh" model-base baseline round3_fp8_model.py --tag reproduced-base
PR5090_CUDA_GRAPH=1 bash "$PR5090_TASK_ROOT/round3/run_sglang_full.sh" model-patch patched round3_fp8_model.py --tag reproduced-patch-graph
```

Run both model arms in both modes for the paired matrix. The baseline suffix and host-budget regression selections are expected to exit nonzero. The complete candidate suites and successful models must exit zero. `summary.json` and raw XML/JSON/log/thermal records separate these results.

Initial setup attempts are retained as failures: an exact-source assertion caught unrelated nightly changes; a relative launcher path needed correction; the default package resolution mixed nvcc 13.4 with runtime headers 13.0; and the first external 13.0 toolchain lacked its separate NVVM package. None of those attempts counts as bug reproduction or successful validation. The final matching toolchain is nvcc/CRT/NVVM 13.0.88, runtime 13.0.96 and CCCL 13.0.85, without disabling compiler compatibility checks. Raw `helper_source_sha256` fields refer to the original PR snapshot used by the reused harness; actual installed nightly-file hashes are recorded separately in the arm/provenance reports.
