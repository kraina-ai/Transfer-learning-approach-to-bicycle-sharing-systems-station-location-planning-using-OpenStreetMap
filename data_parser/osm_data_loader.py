import logging
import os
from json import load

import geopandas as gpd
import pandas as pd
from db_connector.db_definition import db
from db_connector.models import Area, Vector
from osm_data_downloader.src.overpass_queries import QUERIES
from shapely.geometry import mapping, shape
from tqdm import tqdm

from .hex_embedding import generate_hex_vectors
from .hex_generator import generate_hexes


def generate_and_save_vectors(relation_name: str, area_name: str = None) -> None:
    if area_name is None:
        area_name = relation_name

    if Area.get_or_none(Area.name == area_name) is not None:
        logging.warn(f'City {area_name} already exists in the database')
        return

    relation_gdf = load_relation_to_gdf(relation_name, area_name)
    area_shape = relation_gdf.geometry[0]
    elements_gdf = load_relation_elements_to_gdf(relation_name, area_name)

    h3_indexes = generate_hexes(area_shape)
    vectors = generate_hex_vectors(elements_gdf, h3_indexes, area_name)

    shp = area_shape
    area = Area()
    area.name = area_name
    area.lon = shp.centroid.x
    area.lat = shp.centroid.y
    area.shape = mapping(shp)
    area.save()
    area_id = area.id

    data = []
    for hex_id, vector in tqdm(zip(h3_indexes, vectors), desc=f'[{area_name}: {area_id}] Saving hexes vectors', total=len(h3_indexes)):
        v_obj = {
            'area': area_id,
            'hex_id': hex_id
        }
        for i, v in enumerate(vector):
            v_obj[f'v{i}'] = int(v)
        data.append(v_obj)
        if len(data) == 10000:
            Vector.insert_many(data).on_conflict_ignore().execute()
            data.clear()
    Vector.insert_many(data).on_conflict_ignore().execute()


def load_relation_to_gdf(relation_name: str, area_name: str) -> gpd.GeoDataFrame:
    file_path = f'relation/{relation_name}/relation.geojson'
    assert os.path.exists(file_path), f"Error with relation: {relation_name}"
    return gpd.read_file(file_path)
    
def load_relation_elements_to_gdf(relation_name: str, area_name: str) -> gpd.GeoDataFrame:
    directory_path = f'relation/{relation_name}/elements'
    assert os.path.exists(directory_path)
    dicts = []
    for category in tqdm(QUERIES, desc=f'[{area_name}] Loading category objects to GeoDataFrame'):
        if category.name == 'greenery':
            continue
        file_path = f'{directory_path}/{category.name}.geojson'
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                geojson_dict = load(file)
            for feature in geojson_dict["features"]:
                dicts.append({
                    "id": feature["id"],
                    "category": category.name,
                    "geometry": shape(feature["geometry"])
                })
    
    gdf = gpd.GeoDataFrame(dicts)
    return gdf
