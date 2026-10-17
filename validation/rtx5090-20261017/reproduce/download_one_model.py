import sys
from huggingface_hub import snapshot_download

repo, revision, target, *selected = sys.argv[1:]
snapshot_download(repo, revision=revision, local_dir=target,
    allow_patterns=selected or ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "README.md"],
    ignore_patterns=["*.fp16*"], max_workers=2)
print("DOWNLOAD_COMPLETE", repo, flush=True)
