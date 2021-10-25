# from multiprocessing import Manager

import numpy as np
from db_connector.db_definition import db
from db_connector.models import TrainLabel, Vector

# manager = Manager()

HEX_VECTOR_CACHE = {}
HEX_LABEL_CACHE = {}
AREA_EMBEDDING_CACHE = {}

def get_hex_vector(hex_id: str, allow_loading_from_db: bool = False) -> np.ndarray:
    if not hex_id in HEX_VECTOR_CACHE:
        HEX_VECTOR_CACHE[hex_id] = None
        if allow_loading_from_db:
            with db:
                vector_obj = Vector.get_or_none(Vector.hex_id == hex_id)
                if vector_obj is not None:
                    HEX_VECTOR_CACHE[hex_id] = vector_obj.get_vector()
    return HEX_VECTOR_CACHE[hex_id]

def get_hex_label(hex_id: str, allow_loading_from_db: bool = False) -> np.ndarray:
    if not hex_id in HEX_LABEL_CACHE:
        HEX_LABEL_CACHE[hex_id] = None
        if allow_loading_from_db:
            with db:
                label_obj = TrainLabel.get_or_none(TrainLabel.hex_id == hex_id)
                if label_obj is not None:
                    HEX_LABEL_CACHE[hex_id] = label_obj.has_station
    return HEX_LABEL_CACHE[hex_id]

def get_area_embedding_data(area_id: int) -> np.ndarray:
    result = None
    if area_id in AREA_EMBEDDING_CACHE:
        result = AREA_EMBEDDING_CACHE[area_id]
    return result

def save_area_embedding_data(area_id: int, vector: np.ndarray) -> None:
    AREA_EMBEDDING_CACHE[area_id] = vector

def load_hexes_to_cache(area_id: int) -> None:
    for vector_obj in Vector.select().where(Vector.area == area_id):
        hex_id = vector_obj.hex_id
        if not hex_id in HEX_VECTOR_CACHE:
            HEX_VECTOR_CACHE[hex_id] = vector_obj.get_vector()
    
    for label_obj in TrainLabel.select().where(TrainLabel.area == area_id):
        hex_id = label_obj.hex_id
        if not hex_id in HEX_LABEL_CACHE:
            HEX_LABEL_CACHE[hex_id] = label_obj.has_station
    print(f'Hex cache size: {len(HEX_VECTOR_CACHE)} / {len(HEX_LABEL_CACHE)} / {len(AREA_EMBEDDING_CACHE)}')

def clean_cache() -> None:
    HEX_VECTOR_CACHE.clear()
    HEX_LABEL_CACHE.clear()
    AREA_EMBEDDING_CACHE.clear()
