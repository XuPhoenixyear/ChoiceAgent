"""Small environment checks: no model downloads, API calls, or training jobs."""
from __future__ import annotations
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
# Only this short-lived process chooses a GPU. Leave an existing user selection intact.
selected_physical_gpu = None
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        text=True,
    ).strip().splitlines()
    free = [(int(row.split(',')[1]), row.split(',')[0].strip()) for row in rows]
    selected_physical_gpu = max(free)[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = selected_physical_gpu

modules = {
    'torch': 'torch', 'transformers': 'transformers',
    'sentence-transformers': 'sentence_transformers', 'accelerate': 'accelerate',
    'sentencepiece': 'sentencepiece', 'faiss-cpu': 'faiss', 'numpy': 'numpy',
    'scipy': 'scipy', 'pandas': 'pandas', 'scikit-learn': 'sklearn',
    'pyarrow': 'pyarrow', 'fastapi': 'fastapi', 'uvicorn': 'uvicorn',
    'pydantic': 'pydantic', 'pydantic-settings': 'pydantic_settings',
    'streamlit': 'streamlit', 'httpx': 'httpx', 'openai': 'openai',
    'PyYAML': 'yaml', 'python-dotenv': 'dotenv', 'rich': 'rich',
    'tqdm': 'tqdm', 'matplotlib': 'matplotlib', 'ipykernel': 'ipykernel',
    'pytest': 'pytest', 'pytest-asyncio': 'pytest_asyncio',
}
versions = {}
for package, module in modules.items():
    loaded = importlib.import_module(module)
    module_path = getattr(loaded, '__file__', None)
    if module_path is not None:
        assert Path(module_path).resolve().is_relative_to(Path(sys.prefix).resolve()), (package, module_path)
    versions[package] = importlib.metadata.version(package)

import torch
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, Qwen3Config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

assert sys.version_info[:2] == (3, 11)
assert sys.prefix.endswith('/envs/my_rec'), sys.prefix
assert os.environ.get('PYTHONNOUSERSITE') == '1'
assert torch.cuda.is_available(), 'CUDA is not available'
with torch.inference_mode():
    a = torch.arange(64, device='cuda', dtype=torch.float32).reshape(8, 8)
    actual = (a @ a.T).cpu()
    expected = a.cpu() @ a.cpu().T
    assert torch.allclose(actual, expected)
    torch.cuda.synchronize()
cuda_result = {
    'runtime': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
    'physical_gpu_selected_for_this_check': selected_physical_gpu,
    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    'visible_devices_during_check': os.environ.get('CUDA_VISIBLE_DEVICES'),
}
del a, actual, expected
torch.cuda.empty_cache()

# A tiny randomly initialized Qwen3 validates the model implementation without weights.
config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=8,
                    max_position_embeddings=32, bos_token_id=1,
                    eos_token_id=2, pad_token_id=0)
model = AutoModelForCausalLM.from_config(config).eval()
with torch.inference_mode():
    logits = model(torch.tensor([[1, 5, 8]])).logits
assert tuple(logits.shape) == (1, 3, 128)

vectors = np.eye(3, dtype=np.float32)
index = faiss.IndexFlatIP(3)
index.add(vectors)
scores, ids = index.search(vectors[[1]], 1)
assert int(ids[0, 0]) == 1 and np.isclose(scores[0, 0], 1)

with sqlite3.connect(':memory:') as db:
    db.execute('CREATE VIRTUAL TABLE items USING fts5(title)')
    db.executemany('INSERT INTO items(title) VALUES(?)', [('desk organizer',), ('lamp',)])
    hits = db.execute("SELECT title FROM items WHERE items MATCH ? ORDER BY bm25(items)", ('desk',)).fetchall()
    assert hits == [('desk organizer',)]

app = FastAPI()
class Query(BaseModel):
    limit: int = Field(default=3, ge=1, le=10)
@app.post('/check')
def check(query: Query):
    return {'limit': query.limit}
with TestClient(app) as client:
    assert client.post('/check', json={'limit': 3}).json() == {'limit': 3}
    assert client.post('/check', json={'limit': 99}).status_code == 422

report = {
    'status': 'passed', 'python': sys.version.split()[0], 'executable': sys.executable,
    'prefix': sys.prefix, 'sqlite': sqlite3.sqlite_version, 'cuda': cuda_result,
    'checks': ['dependency imports', 'CUDA matrix multiplication',
               'tiny random Qwen3 CPU forward', 'FAISS exact search',
               'SQLite FTS5 BM25', 'FastAPI/Pydantic request validation'],
    'versions': versions,
    'scope': 'Environment validation only; no pretrained model or network API tested.',
}
(ROOT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
