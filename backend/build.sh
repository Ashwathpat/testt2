#!/usr/bin/env bash
# Render build script: install deps + pre-download the ONNX embedding model
# so it's available at runtime without any network calls.
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Pre-downloading ONNX embedding model ==="
python -c "
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath('.')), 'fastembed_cache')
MODEL_NAME = 'intfloat/multilingual-e5-small'

try:
    TextEmbedding.add_custom_model(
        model=MODEL_NAME,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=MODEL_NAME),
        dim=384,
        model_file='onnx/model.onnx',
    )
except ValueError:
    pass

model = TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR, threads=1)
result = list(model.embed(['test embedding']))
print(f'Model downloaded and tested OK. Vector dim: {len(result[0])}')
"

echo "=== Build complete ==="
