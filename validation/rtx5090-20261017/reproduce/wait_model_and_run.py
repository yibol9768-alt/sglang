"""Wait for atomically completed checkpoint files, then run the guarded test."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
spec = {
    "Qwen3-8B-FP8": [("model-00001-of-00002.safetensors", 4952803168), ("model-00002-of-00002.safetensors", 4483005432)],
    "Qwen3-30B-A3B-NVFP4": [(f"model-{i:05d}-of-00004.safetensors", size) for i, size in enumerate([5002279496, 5002724080, 5002036032, 3089670712], 1)],
    "Sana_600M_1024px_diffusers": [
        ("text_encoder/model-00001-of-00002.safetensors", 4988024144),
        ("text_encoder/model-00002-of-00002.safetensors", 240691624),
        ("transformer/diffusion_pytorch_model.safetensors", 2367054536),
        ("vae/diffusion_pytorch_model.safetensors", 1249044836),
    ],
}
name, *command = sys.argv[1:]
model = ROOT / "models" / name
start = time.monotonic()
while True:
    ready = all((model/f).exists() and (model/f).stat().st_size == size for f, size in spec[name])
    if name.startswith("Sana"):
        ready = ready and (model/"model_index.json").exists() and (model/"tokenizer/tokenizer_config.json").exists()
    else:
        ready = ready and (model/"config.json").exists() and (model/"tokenizer_config.json").exists()
        ready = ready and ((model/"tokenizer.json").exists() or ((model/"vocab.json").exists() and (model/"merges.txt").exists()))
    if ready:
        print("CHECKPOINT_READY", name, flush=True)
        break
    allocated = sum(p.stat().st_blocks*512 for p in model.rglob("*") if p.is_file())
    print(json.dumps({"waiting_for": name, "downloaded_gib": round(allocated/1024**3, 2), "elapsed_s": round(time.monotonic()-start)}), flush=True)
    time.sleep(45)
    if time.monotonic()-start > 3*3600:
        raise TimeoutError("Checkpoint download not complete after 3 hours")
raise SystemExit(subprocess.call(command))
