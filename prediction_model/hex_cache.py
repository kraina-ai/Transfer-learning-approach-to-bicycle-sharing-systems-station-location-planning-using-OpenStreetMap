from typing import List
import numpy as np
from db_connector.db_definition import db
from db_connector.models import TrainLabel, Vector
from tqdm import tqdm
import logging


ALL_TRAINED_CACHED = False
AREAS_IDS = set()
HEX_VECTOR_CACHE = {}
HEX_LABEL_CACHE = {}
AREA_EMBEDDING_CACHE = {}
AREA_HEX_IDS_CACHE = {}

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

def get_area_hex_id_data(area_id: int) -> List[str]:
    result = None
    if area_id in AREA_HEX_IDS_CACHE:
        result = AREA_HEX_IDS_CACHE[area_id]
    return result

def save_area_hex_id_data(area_id: int, hexes: List[str]) -> None:
    AREA_HEX_IDS_CACHE[area_id] = hexes

def load_hexes_to_cache(area_id: int) -> None:
    if not area_id in AREAS_IDS:
        logging.debug(f'Loading data to cache for area {area_id}')
        for vector_obj in tqdm(Vector.select().where(Vector.area_id == area_id).iterator(), desc=f"{area_id} vectors"):
            hex_id = vector_obj.hex_id
            if not hex_id in HEX_VECTOR_CACHE:
                HEX_VECTOR_CACHE[hex_id] = vector_obj.get_vector()
        
        for label_obj in tqdm(TrainLabel.select().where(TrainLabel.area_id == area_id).iterator(), desc=f"{area_id} labels"):
            hex_id = label_obj.hex_id
            if not hex_id in HEX_LABEL_CACHE:
                HEX_LABEL_CACHE[hex_id] = label_obj.has_station
        AREAS_IDS.add(area_id)
    logging.debug(f'Hex cache size: {len(HEX_VECTOR_CACHE)} / {len(HEX_LABEL_CACHE)} / {len(AREA_EMBEDDING_CACHE)}')

def load_all_hexes_to_cache() -> None:
    global ALL_TRAINED_CACHED
    if not ALL_TRAINED_CACHED:
        logging.debug(f'Loading all data to cache')
        for vector_obj in tqdm(Vector.select().where(Vector.area_id <= 34).iterator(), desc=f"all vectors"):
            hex_id = vector_obj.hex_id
            if not hex_id in HEX_VECTOR_CACHE:
                HEX_VECTOR_CACHE[hex_id] = vector_obj.get_vector()
        
        for label_obj in tqdm(TrainLabel.select().where(TrainLabel.area_id <= 34).iterator(), desc=f"all labels"):
            hex_id = label_obj.hex_id
            if not hex_id in HEX_LABEL_CACHE:
                HEX_LABEL_CACHE[hex_id] = label_obj.has_station
        ALL_TRAINED_CACHED = True
        AREAS_IDS.update(range(1,35))
    logging.debug(f'Hex cache size: {len(HEX_VECTOR_CACHE)} / {len(HEX_LABEL_CACHE)} / {len(AREA_EMBEDDING_CACHE)}')

def clean_cache() -> None:
    HEX_VECTOR_CACHE.clear()
    HEX_LABEL_CACHE.clear()
    AREA_EMBEDDING_CACHE.clear()
