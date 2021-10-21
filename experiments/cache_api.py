from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pydantic import BaseModel, ValidationError
from typing import Any, List, Union
import numpy as np
from pymongo.mongo_client import MongoClient
from tqdm.auto import tqdm

class RequestModel(BaseModel):
    city_id: int
    sample: bool

app = FastAPI()

async def http422_error_handler(_, exc):
    print(_.json())
    return JSONResponse(
        {"errors": exc.errors()}
    )

app.add_exception_handler(ValidationError, http422_error_handler)

loaded_vectors = {}

def load_data():
    client = MongoClient('mongodb://localhost:27017/')
    # coll_hex = client.osmDataDB.hexesInCitiesFiltered
    coll_hex_cache = client.osmDataDB.hexNeighbourEmbeddingCacheBaseCountCategoryEmbeddingFull
    for h in tqdm(coll_hex_cache.find({ 'city_id': { '$exists': True } }, { "city_id": 1, "has_station": 1, "vector": 1 })):
        city_id = h['city_id']
        has_station = int(h['has_station'])
        vector = h['vector']
        if not city_id in loaded_vectors:
            loaded_vectors[city_id] = {
                1: [],
                0: []
            }
        loaded_vectors[city_id][has_station].append(vector)
    for k, v in loaded_vectors.items():
        loaded_vectors[k][0] = np.array(loaded_vectors[k][0])
        loaded_vectors[k][1] = np.array(loaded_vectors[k][1])
        print(k, loaded_vectors[k][1].shape, loaded_vectors[k][0].shape)

@app.post("/get_data")
async def predict(request: RequestModel):
    stations_array = loaded_vectors[request.city_id][1]
    non_stations_array = loaded_vectors[request.city_id][0]
    if request.sample:
        non_stations_array = non_stations_array[np.random.choice(non_stations_array.shape[0], int(stations_array.shape[0] * 2.5), replace=False), :]
    print(request.city_id, stations_array.shape, non_stations_array.shape)
    return { "stations": stations_array.tolist(), "non_stations": non_stations_array.tolist() }

load_data()