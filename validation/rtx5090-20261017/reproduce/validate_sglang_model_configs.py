"""Validate real tokenizer/config assets before the large weights finish."""
from pathlib import Path
from transformers import AutoTokenizer
from sglang.srt.server_args import ServerArgs
from sglang.multimodal_gen.runtime.server_args import ServerArgs as DiffusionArgs

ROOT = Path(__file__).resolve().parent
text_path = str(ROOT / "models/Qwen3-8B-FP8")
tokenizer = AutoTokenizer.from_pretrained(text_path, local_files_only=True)
prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Calculate 17 * 23."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
assert len(tokenizer.encode(prompt)) < 1024
args = ServerArgs(model_path=text_path, dtype="bfloat16", tp_size=1,
    context_length=1024, mem_fraction_static=0.65, max_running_requests=4,
    chunked_prefill_size=512, attention_backend="triton", enable_deterministic_inference=True,
    disable_cuda_graph=True, cuda_graph_max_bs_decode=4, disable_overlap_schedule=True,
    log_level="info", random_seed=0)
print("REAL_FP8_TOKENIZER_CONFIG_OK", flush=True)
diffusion = DiffusionArgs.from_kwargs(model_path=str(ROOT/"models/Sana_600M_1024px_diffusers"),
    num_gpus=1, component_residency={"dit": "resident", "vae": "resident", "text_encoder": "component-offload"},
    enable_torch_compile=False, warmup_mode="off", attention_backend="sdpa")
print("REAL_SANA_CONFIG_OK", flush=True)
