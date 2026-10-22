# Deterministic FP8 pipeline optimization on RTX 5090, 2026-10-22

For SGLang #44861. The additional production change is one launch parameter:
`num_stages=2` becomes `3` in `apply_fp8_linear_deterministic`. N=64/K=128,
M-only tiles16/32/128, warps4, quantization, accumulation/scaling order, dispatch
and normal-mode behavior remain unchanged. This is not a new kernel or public option.

## Runtime and comparison

One RTX5090, TP1, WSL2, 400W. The persistent NVML guard stops only our test
process group at72C and fails closed if telemetry goes stale. Passive cooling
to<=55C occurs outside timed requests. Maximum across all ten jobs:
66C. No power/fan settings or existing services were changed.

Official SGLang0.5.23.dev20261020+g55b2aef7ad, Torch2.14.1, Triton3.8.0, FlashInfer0.7.2, sglang-kernel0.4.10; complete files, normal imports

Model: `nvidia/Qwen3-8B-FP8@2cebc4c89e25abc17668c81b01dceaf3d8b914d5`. BF16 activations, automatic FP8 KV, deterministic
inference, Triton attention, overlap off; context6144, batch ceiling4,
prefill chunk8192. The production PR parent is `5f90880cc296ce831f9f1914a48285e17c848100`; the validated optimization commit is `6a7695ddc4630c5714a5b1e072550f5f3e58bd49`.
"Current" is its two-stage path, "candidate" the stage-only change, and
"upstream" the unmodified FP8 files from the same official nightly. Complete runtime source files are
selected before process startup, with normal imports and no runtime function injection.
Per-run source manifests retain the installed file hashes.

## Exact output preservation

- Complete existing SM120 test file: **137 passed**.
- Additional paired tensor comparisons: **288/288 exact**, BF16/FP16, two seeds,
  eight N/K shapes including real projections and K tails, M tile boundaries,
  bias and strided inputs. Both current-PR equivalence and full/cache-suffix
  equivalence are checked.
- Real-model cache comparisons: **18 paired cases exact** across
  two Graph boots per arm and one eager boot per arm. Prefixes128/1024/4096,
  batches1/4/2, two repetitions; cold/cached selected-token logprobs and warm
  repeats have zero difference. Cross-arm score arrays also match exactly.
- **180 complete fixed-work generated outputs** match between current and candidate,
  including token IDs in every warmup and measured repetition.
- **128/128 full GSM8K answer texts and token sequences match exactly**,120/128
  correct in each arm, no missing final answers,512-token cap. The questions
  reuse earlier test rows32–159; this is an output-preservation regression,
  not an independent quality evaluation or an accuracy-improvement claim.

## End-to-end timing

32 forced output tokens per request, one warmup and five measured samples per
cell per engine boot, cache flushed before each request. PR-arm Graph boot
order is A(current), B(candidate +quality), B(candidate), A(current +quality).
The table pools the ten measured samples per PR arm. Raw samples and each
boot's median are in `summary.json`; in particular the2048/1 cell has visible
between-boot variation. The upstream reference and each eager arm have only
one boot/five measured samples, so they are more limited controls.

| Prefix / batch | Graph current,s | Graph candidate,s | Reduction | Graph upstream,s | Eager current → candidate,s |
|---|---:|---:|---:|---:|---:|
| 128 / 1 | 0.3119 | 0.2796 | 10.3% | 0.2684 | 1.4868 → 1.4839 |
| 128 / 4 | 0.3539 | 0.3251 | 8.1% | 0.3322 | 1.5320 → 1.5275 |
| 2048 / 1 | 0.5314 | 0.5143 | 3.2% | 0.4766 | 1.5629 → 1.5642 |
| 2048 / 4 | 0.8900 | 0.8688 | 2.4% | 0.7953 | 2.0843 → 2.0747 |

These are bounded observations on one model/device. The improvement does not
make deterministic mode universally faster than the normal upstream dispatch.
The upstream Graph reference retains the original selected-logprob cache drift.
Torch.compile, multi-GPU, SM121 and other model/device performance were not tested.

The component probe uses a256MiB eviction before each CUDA-event-timed graph
call; eviction is outside timed events. It separates quantization from GEMM
and rejects schedules that change outputs. These microbenchmarks are exploratory
and do not substitute for the end-to-end measurements above.

## Reproduce

Use the pinned round3 official-nightly installation and CUDA13.0 toolchain
from the parent artifact directory, plus its model and pinned128-question dataset.
Copy `candidate/` into `$PR5090_TASK_ROOT/round4/candidate`, the reproduction
scripts into `round4/`, `cool_gpu.py` and `gpu_guard.py` into `gpu_payload/`,
and `fp8_model.py` into `gpu_payload/round4_fp8_model.py`. Preserve the round3
baseline/patched snapshots; patched there is the immutable two-stage PR parent.
The existing test file belongs in `round3/test_sm120_fp8_linear.py`.

```bash
export PR5090_TASK_ROOT=/absolute/path/to/isolated/pr5090
bash "$PR5090_TASK_ROOT/round4/run_fp8.sh" equality candidate "$PR5090_TASK_ROOT/round4/validate_pipeline.py"
bash "$PR5090_TASK_ROOT/round4/run_fp8.sh" native-tests candidate "$PR5090_TASK_ROOT/round4/run_tests.py"
PR5090_CUDA_GRAPH=1 bash "$PR5090_TASK_ROOT/round4/run_fp8.sh" graph-current current "$PR5090_TASK_ROOT/gpu_payload/round4_fp8_model.py" --tag r4-graph-current
PR5090_CUDA_GRAPH=1 bash "$PR5090_TASK_ROOT/round4/run_fp8.sh" graph-candidate-1 candidate "$PR5090_TASK_ROOT/gpu_payload/round4_fp8_model.py" --tag r4-graph-candidate-1 --quality 128
```

Repeat candidate then
current (with the128-question check) to complete ABBA; run the upstream arm
in Graph mode and both PR arms with `PR5090_CUDA_GRAPH=0` for the remaining
controls. Every job must exit zero; guard interruptions invalidate a run.
`analyze_results.py` checks paired JSONs, XML and all ten thermal reports before
writing the final summary. `manifest.json` hashes every published file.

All GPU runs above succeeded. The first local pre-commit attempt used an old
system Python for a repository hook; rerunning with the existing Python3.12
environment on PATH passed all applicable hooks. This setup issue does not
affect GPU results. AI assistance was used for diagnosis, experiments and the update.
Local results and external reproductions do not constitute upstream CI or maintainer approval.
