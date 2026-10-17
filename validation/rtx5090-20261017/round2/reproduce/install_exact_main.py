"""Install the official wheel for the PR's exact base into an isolated env."""
import json
import os
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parent.parent
UV = Path(shutil.which('uv') or '/home/user/.local/bin/uv')
ENV_ROOT = ROOT / 'envs/vllm-main'
PYTHON = ENV_ROOT / 'bin/python'
env = dict(os.environ, UV_CACHE_DIR=str(ROOT / 'tools/uv-main-cache'), UV_HTTP_TIMEOUT='180')
if not PYTHON.exists():
    subprocess.run([str(UV), 'venv', '--python', '/usr/bin/python3', str(ENV_ROOT)], env=env, check=True)
url = 'https://wheels.vllm.ai/ce49c47cbb241b59a32ef110079a5261dbd08e49/vllm-0.32.0rc1.dev323%2Bgce49c47cb-cp38-abi3-manylinux_2_28_x86_64.whl'
subprocess.run([str(UV), 'pip', 'install', '--python', str(PYTHON), '--link-mode=copy',
                '--torch-backend=cu130', url, 'pytest', 'nvidia-ml-py', 'tblib',
                'pytest-asyncio', 'pytest-subtests'], env=env, check=True)
freeze = subprocess.check_output([str(UV), 'pip', 'freeze', '--python', str(PYTHON)], env=env, text=True)
(ROOT / 'gpu_payload/exact-main-freeze.txt').write_text(freeze)
(ROOT / 'gpu_payload/exact-main-install.json').write_text(json.dumps({'base_commit':
    'ce49c47cbb241b59a32ef110079a5261dbd08e49', 'wheel_url': url,
    'environment': str(ENV_ROOT), 'status': 'installed'}, indent=2))
print('EXACT_MAIN_INSTALLED', flush=True)
