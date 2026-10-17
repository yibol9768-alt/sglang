"""Run a pretrained SANA pipeline with component offload in both budget arms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["baseline", "patched"])
    args = ap.parse_args()
    nested = os.environ.get("PR5090_NESTED_HOST_BUDGET") == "1"
    tag = args.arm + ("-nested" if nested else "")
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
    from PIL import Image
    import numpy as np
    started = time.monotonic()
    generator = DiffGenerator.from_pretrained(
        model_path=str(ROOT / "models/Sana_600M_1024px_diffusers"), num_gpus=1,
        component_residency={"dit": "layerwise-offload" if nested else "resident", "vae": "resident", "text_encoder": "resident"},
        dit_layerwise_resident_layers=27 if nested else 0,
        dit_layerwise_residency_lifetime="permanent" if nested else "forward",
        enable_torch_compile=False, warmup_mode="off", attention_backend="torch_sdpa",
    )
    loaded = time.monotonic() - started
    try:
        (ROOT / "sana-images").mkdir(exist_ok=True)
        cases = []
        for trial, (prompt, seed) in enumerate([
            ("A small red wooden cabin beside a calm lake, green pine trees and distant mountains, watercolor illustration.", 42),
            ("A ceramic teapot and two blue cups on a wooden table, soft window light, still life photograph.", 17),
        ]):
            started = time.perf_counter()
            result = generator.generate({"prompt": prompt, "height": 512, "width": 512,
                "num_inference_steps": 12, "guidance_scale": 4.5, "seed": seed,
                "save_output": True, "output_path": str(ROOT / "sana-images"),
                "output_file_name": f"{tag}-{trial}.png"})
            duration = time.perf_counter() - started
            assert result is not None
            if isinstance(result, list):
                assert len(result) == 1
                result = result[0]
            image_path = Path(result.output_file_path)
            pixels = np.asarray(Image.open(image_path).convert("RGB"))
            assert pixels.shape == (512, 512, 3)
            assert pixels.std() > 1, "Blank or degenerate output image"
            cases.append({"prompt": prompt, "seed": seed, "seconds": duration,
                "generation_seconds": result.generation_time, "peak_memory_mb": result.peak_memory_mb,
                "image_path": str(image_path), "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "pixel_mean": float(pixels.mean()), "pixel_std": float(pixels.std())})
            print("REAL_SANA_CASE", json.dumps(cases[-1]), flush=True)
        report = {"arm": args.arm, "nested_cgroup": nested, "model": "Efficient-Large-Model/Sana_600M_1024px_diffusers",
                  "autoencoder_checkpoint_fix": os.environ.get("PR5090_VAE_CHECKPOINT_FIX") == "1",
                  "revision": "28f3af7689de15f3883d5863059a2fca0aa9b829", "load_seconds": loaded,
                  "mode": "Full SGLang Diffusion pipeline; eager execution; main-source host budget; " + ("one streamed DiT block in live nested cgroup" if nested else "GPU-resident components"),
                  "cases": cases}
        (ROOT / f"sana-real-{tag}.json").write_text(json.dumps(report, indent=2))
    finally:
        generator.shutdown()

if __name__ == "__main__":
    main()
