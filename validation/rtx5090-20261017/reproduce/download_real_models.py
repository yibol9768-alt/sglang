"""Fetch public pretrained checkpoints at fixed revisions into task storage."""
import json
from pathlib import Path
import time

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS = [
    ("RedHatAI/Qwen3-30B-A3B-NVFP4", "a4df9f017835830cbb588cfe72bc4e50294183c0"),
    ("nvidia/Qwen3-8B-FP8", "2cebc4c89e25abc17668c81b01dceaf3d8b914d5"),
]
for repo, revision in MODELS:
    start = time.monotonic()
    target = ROOT / "models" / repo.split("/")[-1]
    print(json.dumps({"event": "start", "repo": repo, "revision": revision}), flush=True)
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "README.md"],
        max_workers=2,
    )
    print(json.dumps({"event": "complete", "repo": repo, "seconds": time.monotonic()-start, "path": str(target)}), flush=True)
