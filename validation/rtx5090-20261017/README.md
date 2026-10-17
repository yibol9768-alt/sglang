# RTX 5090 validation records — 2026-10-17

Supporting evidence for three independent bug fixes: vLLM compressed-tensors NVFP4 weight offload, SGLang diffusion ancestor-cgroup headroom, and SGLang SM12x deterministic ModelOpt FP8. This evidence branch is separate from each production PR diff.

## Scope and limitations

- One RTX 5090, TP=1, 400W limit, stop threshold 72°C. Linux/WSL2, Python 3.12. Private compiled runtimes: vLLM 0.31.0 and SGLang 0.5.21; full package versions are in `reproduce/env-*.freeze.txt`.
- The changed main-source Python definitions were selected inside private environments. Affected baseline definitions were checked against main using AST hashes. This is **not a rebuild of complete main** and does not replace upstream CI. Baselines: vLLM `ce49c47cbb241b59a32ef110079a5261dbd08e49`; SGLang `360382454170073d54c202745e9884ee51f72fa3`.
- Only workspace/home path strings were replaced with `/path/to/` placeholders for publication. Numeric measurements and model outputs are unchanged. The guard and environment preparation scripts require adapting the documented task/CUDA/header/library paths; they are not a generic installer.
- AI assistance was used for diagnosis, implementation, validation and write-up. The human contributor reviewed the local material and authorized submission.

## Model revisions

| Model | Pinned revision |
|---|---|
| `RedHatAI/Qwen3-30B-A3B-NVFP4` | `a4df9f017835830cbb588cfe72bc4e50294183c0` |
| `nvidia/Qwen3-8B-FP8` | `2cebc4c89e25abc17668c81b01dceaf3d8b914d5` |
| `Efficient-Large-Model/Sana_600M_1024px_diffusers` | `28f3af7689de15f3883d5863059a2fca0aa9b829` |

Checkpoints are not included. Download pinned public snapshots into the reproduction directory's `models/<repository-name>/` before running. GSM8K uses the first 32 examples from test.jsonl at `openai/grade-school-math` revision `3101c7d5072418e28b9008a6636bde82a006892c`; the selected data and its MIT license are included.

## vLLM NVFP4

`evidence/nvfp4-real-comparison.json` summarizes six full-model starts: resident, original offload, patched offload, patched CUDA Graph, and original/patched repeated starts with a 32-question check. The first four used `reproduce/nvfp4_real_model_initial.py`; the repeated pair used `reproduce/nvfp4_real_model.py` with additional scale hashes and evaluation. Under expert-only `cpu_offload_gb=1`, the fix restores 1,107,296,256 main expert bytes to UVA and reduces CUDA allocated bytes by exactly that amount. All 96 main weight hashes agree; all 384 scale hashes agree in the repeated pair.

Real-model token/logprob equivalence is **not established**. The unmodified build itself changes full token sequences between starts (only 3/8 identical); the paired GSM8K check is 29/32 versus 30/32, including missing/truncated final-answer effects. This is a small smoke check, not evidence of general accuracy improvement. Fixed eager timing adds approximately 11–15% wall time when the weights really remain on the host.

The native regression is collected with normal repository fixtures by `run_vllm_regression_pytest.py`, selecting the baseline or candidate `device_loading_context`; original 2 failed, candidate 2 passed. It tests real CUDA offloader tensors under UVA enabled/disabled and a resident control. Full pretrained runs use UVA; complete non-UVA model coverage is not claimed.

## SGLang ModelOpt FP8

`sglang-real-*.json` records eager and CUDA Graph cold/warm selected-logprob comparisons. Original differences become zero in all three measured branches. `native-fp8-pytest-final.log` records the final 12-case method regression; baseline is 4 failed / 8 passed. GSM8K first 32 is 28/32 versus 30/32; this is a small smoke check. Fixed Graph timing is 0.254s versus 0.279s, approximately 10% slower. The deterministic reduction uses fixed 32×64×128 tiling; normal-mode dispatch is unchanged.

## SGLang host budget

`host-budget-*.log` covers the four added v1/v2 cases (baseline 4 failed / 32 passed; candidate 36 passed). `host-budget-live-cgroup.json` uses an actual delegated cgroup v2 hierarchy with 512MiB allocated in a sibling. `sana-real-*-nested.json`, `sana-cgroup-*.json` and `sana-nested-*-trim.log` record the final real-SANA nested-cgroup comparison: one streamed DiT block, 27 resident blocks, 41,292,288 bytes pinned by the old budget versus pageable storage under the fix, two pixel-identical outputs, zero OOM/OOM-kill events. Outer reclaim/limit events are reported as recorded and are not claimed to be zero.

Both SANA arms use the **same checkpoint-loading prerequisite** in `prepare_autoencoder_arm.py` and its source snapshots. Its root fix already exists in [SGLang #44690](https://github.com/sgl-project/sglang/pull/44690); it is not proposed as another PR or credited as a new independent fix. No production OOM was deliberately triggered.

## Reproduction entry points

After creating isolated runtimes and adapting task paths in `run_guarded_model.sh`, the actual guarded entry points are:

```bash
bash run_guarded_model.sh vllm run_vllm_regression_pytest.py baseline
bash run_guarded_model.sh vllm run_vllm_regression_pytest.py patched
bash run_guarded_model.sh vllm nvfp4_real_model.py resident
bash run_guarded_model.sh vllm nvfp4_real_model.py baseline
bash run_guarded_model.sh vllm nvfp4_real_model.py patched
PR5090_CUDA_GRAPH=1 bash run_guarded_model.sh vllm nvfp4_real_model.py patched
bash run_guarded_model.sh vllm nvfp4_real_model.py baseline --tag baseline-repeat --gsm8k
bash run_guarded_model.sh vllm nvfp4_real_model.py patched --tag patched-repeat --gsm8k
bash run_guarded_model.sh sglang run_sglang_regression_pytest.py baseline
bash run_guarded_model.sh sglang run_sglang_regression_pytest.py patched
bash run_guarded_model.sh sglang sglang_real_model.py baseline
bash run_guarded_model.sh sglang sglang_real_model.py patched
PR5090_CUDA_GRAPH=1 bash run_guarded_model.sh sglang sglang_real_model.py baseline
PR5090_CUDA_GRAPH=1 bash run_guarded_model.sh sglang sglang_real_model.py patched
bash run_guarded_model.sh sglang sglang_gsm8k_smoke.py baseline
bash run_guarded_model.sh sglang sglang_gsm8k_smoke.py patched
```

The vLLM native wrapper also expects a checkout with the changed test and standard fixtures at the task's `selected-vllm/`. The pure CPU host-budget entry is `test_host_budget_linux.py --baseline` or without the flag, using the SGLang private interpreter. Nested SANA additionally requires a **dedicated** systemd user service named `codex-pr5090-sana-*.service`, `Delegate=memory`, `MemoryMax=8G`, `MemorySwapMax=256M`, and `PR5090_MEMORY_SCOPED=1`; `live_sana_cgroup.py` rejects other service names. It must not be run inside an existing user research service.

All applicable pre-commit hooks passed for the actual production diffs. Search snapshots, exact targeted PR heads and the limitations of duplicate-work searches are retained in the evidence JSON. No claim is made that those searches see unpublished work.
