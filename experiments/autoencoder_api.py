from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pydantic import BaseModel, ValidationError
from typing import Any, List, Union
import numpy as np
from autoencoder.autoencoder_model import Autoencoder

class RequestModel(BaseModel):
    resolution: int
    size: int
    filtered: bool
    # vector: Any
    vector: List[float]

app = FastAPI()

async def http422_error_handler(_, exc):
    print(_.json())
    return JSONResponse(
        {"errors": exc.errors()}
    )

app.add_exception_handler(ValidationError, http422_error_handler)

# models = {
#     9: 'weights/9_PerCategoryShapeAnalyzerEmbedding_5702_300_50000_',
#     10: 'weights/10_PerCategoryShapeAnalyzerEmbedding_5702_300_50000_',
#     11: 'weights/11_PerCategoryShapeAnalyzerEmbedding_5702_300_50000_'
# }

loaded_models = {}

def get_model_file(res, size):
    return f'weights/dropout_100_{res}_PerCategoryFilteredShapeAnalyzerEmbedding_888_{size}_100000_'

@app.post("/predict")
async def predict(request: RequestModel):
    # print(request)
    if not (request.resolution, request.size, request.filtered) in loaded_models:
        print(f'Loading autoencoder for res {request.resolution} and size {request.size} (Filtered: {request.filtered})')
        if request.filtered:
            model = Autoencoder(888, request.size)
            model.load_weights(get_model_file(request.resolution, request.size)).expect_partial()
            loaded_models[(request.resolution, request.size, request.filtered)] = model
        else:
            model = Autoencoder(5702, request.size)
            model.load_weights(f'weights/{request.resolution}_PerCategoryShapeAnalyzerEmbedding_5702_300_50000_').expect_partial()
            loaded_models[(request.resolution, request.size, request.filtered)] = model
    model = loaded_models[(request.resolution, request.size, request.filtered)]
    embedding_tensor = model.get_embedding(np.array(request.vector).reshape((1,-1)))
    embedding_vector = np.array(embedding_tensor).reshape((-1,))
    return embedding_vector.tolist()