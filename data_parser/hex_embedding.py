from typing import List

import numpy as np
import shapely
import geopandas as gpd
from h3 import h3
from osm_data_downloader.src.overpass_queries import QUERIES
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def generate_hex_vectors(osm_gdf: gpd.GeoDataFrame, hex_ids: List[str], area_name: str) -> np.ndarray:
    categories = sorted([query.name for query in QUERIES if query.name != 'greenery'])
    h3_gdf = _generate_polygons_gdf(hex_ids)

    result = np.zeros((len(hex_ids), len(categories)))
    for cat_idx, category in enumerate(categories):
        intersections = gpd.sjoin(osm_gdf.loc[osm_gdf.category == category], h3_gdf, how="inner", predicate='intersects')
        intersections_grouped = intersections.groupby(['hex_id']).count()
        for hex_idx, hex_id in tqdm(enumerate(hex_ids), total=len(hex_ids), desc = f'[{area_name}] [{category}] Embedding hexes into vectors'):
            if hex_id in intersections_grouped.index:
                number = intersections_grouped.loc[hex_id].category
                result[hex_idx, cat_idx] = number
    return result

def _generate_polygons_gdf(hex_ids: List[str]) -> gpd.GeoDataFrame:
    hex_polygons = []
    for hex_id in hex_ids:
        polygons = h3.h3_set_to_multi_polygon([hex_id], geo_json=True)
        outlines = [loop for polygon in polygons for loop in polygon]
        polyline = [outline + [outline[0]] for outline in outlines][0]
        polygon = shapely.geometry.Polygon(polyline)
        hex_polygons.append({
            "hex_id": hex_id,
            "geometry": polygon
        })
    return gpd.GeoDataFrame(hex_polygons)

