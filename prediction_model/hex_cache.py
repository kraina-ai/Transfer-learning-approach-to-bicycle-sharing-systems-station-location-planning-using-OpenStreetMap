import numpy as np
from db_connector.db_definition import db as sqlite_db
from db_connector.models import TrainLabel, Vector

HEX_VECTOR_CACHE = {}
HEX_LABEL_CACHE = {}

def get_hex_vector(hex_id: str, allow_loading_from_db: bool = False) -> np.ndarray:
    if not hex_id in HEX_VECTOR_CACHE:
        HEX_VECTOR_CACHE[hex_id] = None
        if allow_loading_from_db:
            with sqlite_db:
                vector_obj = Vector.get_or_none(Vector.hex_id == hex_id)
                if vector_obj is not None:
                    HEX_VECTOR_CACHE[hex_id] = vector_obj.get_vector()
    return HEX_VECTOR_CACHE[hex_id]

def get_hex_label(hex_id: str, allow_loading_from_db: bool = False) -> np.ndarray:
    if not hex_id in HEX_LABEL_CACHE:
        HEX_LABEL_CACHE[hex_id] = None
        if allow_loading_from_db:
            with sqlite_db:
                label_obj = TrainLabel.get_or_none(TrainLabel.hex_id == hex_id)
                if label_obj is not None:
                    HEX_LABEL_CACHE[hex_id] = label_obj.has_station
    return HEX_LABEL_CACHE[hex_id]

def load_hexes_to_cache(area_id: int) -> None:
    # with lock:
    # with sqlite_db:
    for vector_obj in Vector.select().where(Vector.area == area_id):
        hex_id = vector_obj.hex_id
        if not hex_id in HEX_VECTOR_CACHE:
            HEX_VECTOR_CACHE[hex_id] = vector_obj.get_vector()
    
    for label_obj in TrainLabel.select().where(TrainLabel.area == area_id):
        hex_id = label_obj.hex_id
        if not hex_id in HEX_LABEL_CACHE:
            HEX_LABEL_CACHE[hex_id] = label_obj.has_station
    print(f'Hex cache size: {len(HEX_VECTOR_CACHE)} / {len(HEX_LABEL_CACHE)}')

def clean_cache() -> None:
    HEX_VECTOR_CACHE.clear()
    HEX_LABEL_CACHE.clear()