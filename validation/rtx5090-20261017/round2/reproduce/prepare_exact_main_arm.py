"""Apply complete PR/base Python files to the matching official base wheel."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sysconfig

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('arm', choices=['baseline', 'patched'])
args = parser.parse_args()
version = importlib.metadata.version('vllm')
assert version == '0.32.0rc1.dev323+gce49c47cb', version
assert importlib.metadata.version('torch').startswith('2.14.0')
assert importlib.metadata.version('flashinfer-python') == '0.7.2'
source = ROOT / 'full-source' / f'vllm-main-{args.arm}' / 'vllm'
target = Path(sysconfig.get_paths()['purelib']) / 'vllm'
files = ['model_executor/layers/quantization/base_config.py',
    'model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py',
    'model_executor/model_loader/utils.py']
hashes = {}
for relative in files:
    content = (source / relative).read_bytes()
    destination = target / relative
    temporary = destination.with_suffix('.pr5090-tmp')
    temporary.write_bytes(content)
    temporary.replace(destination)
    hashes[relative] = hashlib.sha256(content).hexdigest()
report = {'arm': args.arm, 'vllm': version,
    'torch': importlib.metadata.version('torch'), 'flashinfer': importlib.metadata.version('flashinfer-python'),
    'files': hashes}
(ROOT / f'exact-main-arm-{args.arm}.json').write_text(json.dumps(report, indent=2))
print('EXACT_MAIN_ARM', json.dumps(report), flush=True)
