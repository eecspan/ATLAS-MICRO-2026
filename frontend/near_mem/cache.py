"""Exact whole-graph reuse; no component latency or DRAM-state approximation."""
import hashlib
import json
from pathlib import Path
from threading import Lock

from .sim import hardware
from .sim import ROOT, dump


def configuration_key(signature):
    inputs = {k:signature[k] for k in ('workload','configuration')}
    return hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()


def task_digest(folder, cores):
    """Hash everything the native worker consumes, except output directory names."""
    operators = json.loads((folder/'operators.yaml').read_text())
    digest = hashlib.sha256((folder/'placement.yaml').read_bytes())
    for original in operators['operator']:
        op = dict(original); prefix = Path(op.pop('file_prefix'))
        digest.update(json.dumps(op,sort_keys=True).encode())
        for core in range(cores):
            payload = (prefix/f'core_{core}.yaml').read_bytes()
            digest.update(len(payload).to_bytes(8,'little')); digest.update(payload)
    return digest.hexdigest()


class ExecutionCache:
    def __init__(self, chip, root, provenance):
        arch,_,paths = hardware(chip); self.cores = arch['core_num']
        keys = set(map(str,paths)) | {str(ROOT/'frontend/near_mem/run.py')}
        self.identity = {k:v for k,v in provenance.items() if k in keys or k.startswith('binary:')}
        self.ready = {}; self.legacy = {}; self.lock = Lock()
        # Older runs have source-keyed signatures. Reuse them only after their
        # actual emitted tasks match byte for byte and backend identities match.
        for signature_file in sorted(Path(root).glob('*/signature.json')):
            folder = signature_file.parent
            if not (folder/'result.json').exists():
                continue
            signature = json.loads(signature_file.read_text())
            if any(signature['sha256'].get(k) != v for k,v in self.identity.items()):
                continue
            stamp = folder/'execution.json'
            if stamp.exists():
                saved = json.loads(stamp.read_text())
                if saved['backend'] == self.identity:
                    self.ready[saved['tasks_sha256']] = folder
            else:
                self.legacy.setdefault(configuration_key(signature),[]).append(folder)

    def get(self, digest, config_key):
        with self.lock:
            for folder in self.legacy.pop(config_key,[]):
                if not (folder/'operators.yaml').exists():
                    continue
                actual = task_digest(folder,self.cores)
                self._save(actual,folder)
            source = self.ready.get(digest)
            if source is not None:
                return json.loads((source/'result.json').read_text()),str(source)
        return None,None

    def _save(self, digest, folder, source=None):
        dump(folder/'execution.json',dict(tasks_sha256=digest,backend=self.identity,reused_from=source))
        self.ready[digest] = folder

    def put(self, digest, folder, source=None):
        with self.lock:
            self._save(digest,folder,source)
