import os
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

MODEL_NAME = "intfloat/multilingual-e5-small"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fastembed_cache")

print(f"Pre-downloading {MODEL_NAME} to {CACHE_DIR} during build step...")

try:
    TextEmbedding.add_custom_model(
        model=MODEL_NAME,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=MODEL_NAME),
        dim=384,
        model_file="onnx/model.onnx",
    )
except ValueError:
    pass

# Force download and cache
model = TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR, threads=1)
print("Model pre-downloaded successfully! ✅")
