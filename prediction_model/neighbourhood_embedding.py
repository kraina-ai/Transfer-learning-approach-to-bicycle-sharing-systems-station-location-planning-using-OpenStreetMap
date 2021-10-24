from typing import List

import numpy as np
from db_connector.db_definition import db as sqlite_db
from db_connector.models.vector import Vector
from h3 import h3
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from .hex_cache import get_hex_vector
from .model_parameters import NEIGHBOURHOOD


def generate_neighbourhood_vector(hex_ids: List[str], additional_desc: str = None, disable_tqdm = True) -> np.ndarray:
    result = np.zeros((len(hex_ids), 20))
    desc = f"Embedding neighbours hexes"
    if additional_desc is not None:
        desc += f" [{additional_desc}]"
    hexes_vectors = process_map(_embed_single_hex, hex_ids, desc=desc, chunksize = 1, disable = disable_tqdm)
    for idx, hex_vector in enumerate(hexes_vectors):
        result[idx, :] = hex_vector
    return result

def _embed_single_hex(hex_id: str) -> np.ndarray:
    vectors = [get_hex_vector(hex_id)]
    for k in range(1, NEIGHBOURHOOD+1):
        vectors.append(_average_neighbours_level(hex_id, k))
    vector = _calculate_embedding(np.array(vectors))
    return vector

def _average_neighbours_level(hex_id: str, k: int) -> np.ndarray:
    indexes = h3.hex_ring(hex_id, k)
    vectors = []
    for h in indexes:
        hex_vector = get_hex_vector(h)
        if hex_vector is not None:
            vectors.append(hex_vector)
    if len(vectors) > 0:
        vector = np.average(np.array(vectors), axis=0).reshape((20,))
    else:
        vector = np.zeros(shape=(20,), dtype='float64')
    return vector

def _calculate_embedding(vectors: np.ndarray) -> np.ndarray:
    # average diminishing squared neighbour embedding
    weights = [1 / ((n + 1)**2) for n in range(len(vectors))]
    return np.average(vectors, axis=0, weights=weights)
