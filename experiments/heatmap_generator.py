from pymongo import ASCENDING, GEOSPHERE, MongoClient
import pandas as pd
from alive_progress import alive_bar
from shapely.geometry import Point, mapping
from keplergl import KeplerGl
import shapely
import json
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import geopandas as gpd
from h3 import h3
import math
import sklearn
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

from neighbour_embedding_methods import AverageDiminishingNeighbourEmbedding, AverageDiminishingSquqredNeighbourEmbedding, AverageNeighbourEmbedding, ConcatenateNeighbourEmbedding
from custom_distance_metric import DistanceMetric
from embedding_methods import BaseCountCategoryEmbedding, BaseShapeAnalyzerEmbedding, PerCategoryShapeAnalyzerEmbedding300
from embedding_methods import PerCategoryFilteredShapeAnalyzerEmbedding300

def generate_xy(cites_ids, resolution, inbalance_ratio, neighbours, neighbour_embedding_cls, embedding_cls, id=None):
    client = MongoClient('mongodb://localhost:27017/')
    coll_hex = client.osmDataDB.hexesInCitiesFiltered
    relation_embedder = embedding_cls()
    relation_embedder.fit({ 'city_id': { '$in': cites_ids } })
    all_stations = [hex for hex in coll_hex.find({ 'city_id': { '$in': cites_ids }, 'has_station': True, 'resolution': resolution })]
    stations_length = len(all_stations)
    non_stations_cursor = coll_hex.aggregate([
        { '$match': { 'city_id': { '$in': cites_ids }, 'has_station': False, 'resolution': resolution } },
        { '$sample': { 'size': stations_length * inbalance_ratio } }
    ])
    non_stations = [hex for hex in non_stations_cursor]
    hex_id_list =  [h['hex_id'] for h in all_stations + non_stations]
    y = np.array([1] * stations_length + [0] * len(non_stations))
    # metric = DistanceMetric()
    # metric.fit({ 'city_id': { '$in': cites_ids } })
    y = np.array([1] * stations_length + [0] * len(non_stations))
    embedder = neighbour_embedding_cls(relation_embedder)
    vectors_list = [embedder.get_embedding(h, neighbours, relation_embedder) for h in tqdm(all_stations + non_stations, desc=f'[{id}] Embedding hexes {embedding_cls.__name__}')]
    # vectors = np.stack(vectors_list, axis=0)
    vectors = np.array(vectors_list)
    # vectors[np.isfinite(vectors) == True] = 0
    try:
        vectors[np.isnan(vectors) == True] = 0
    except:
        print(vectors)
    return vectors, y, hex_id_list

from functools import partial
from tqdm.contrib.concurrent import process_map 

INBALANCE_RATIOS = [2.5]
RESOLUTION = 11
NEIGHBORS = 5
CLASSIFIER = RandomForestClassifier
NEIGHBORS_EMBEDDING_CLASS = AverageDiminishingSquqredNeighbourEmbedding
EMBEDDING_CLASS = BaseCountCategoryEmbedding
ITERATIONS = 100

def _embed(hexes):
    relation_embedder = EMBEDDING_CLASS()
    relation_embedder.fit({}, filtered = False)
    embedder = NEIGHBORS_EMBEDDING_CLASS(relation_embedder)
    return [embedder.get_embedding(h, NEIGHBORS, relation_embedder, check_if_exists=False) for h in hexes]
    # return 0

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_x(hexes, neighbours, neighbour_embedding_cls, embedding_cls, id=None):
    relation_embedder = embedding_cls()
    relation_embedder.fit({}, filtered = False)
    embedder = neighbour_embedding_cls(relation_embedder)
    # _embed = partial(embedder.get_embedding, no_rings = neighbours, embedding_cls = relation_embedder, check_if_exists = False)
    vectors_chunk_lists = process_map(_embed, list(chunks(hexes, 1000)), max_workers=8)
    # vectors_list = [embedder.get_embedding(h, neighbours, relation_embedder, check_if_exists=False) for h in tqdm(hexes, desc=f'[{id}] Embedding hexes {embedding_cls.__name__}')]
    vectors = np.array([v for v_list in vectors_chunk_lists for v in v_list])
    try:
        vectors[np.isnan(vectors) == True] = 0
    except:
        print(vectors)
    return vectors

def normalize_x(x):
    new_array = np.zeros(x.shape)
    # print(x.shape, new_array.shape)
    for i in tqdm(range(x.shape[1])):
        vector = x[:,i].reshape((-1,1))
        scaler = MinMaxScaler()
        new_array[:,i] = scaler.fit_transform(vector).reshape((-1,))
    return new_array

def check_hex(h):
    global shp
    polygons = h3.h3_set_to_multi_polygon([h], geo_json=False)
    outlines = [loop for polygon in polygons for loop in polygon]
    polyline = [outline + [outline[0]] for outline in outlines][0]
    polygon = shapely.geometry.Polygon(polyline)
    if shp.intersects(polygon):
        return {'hex_id': h, 'geometry': polygon}
    return None

client = MongoClient('mongodb://localhost:27017/')
db = client.osmDataDB
coll_areas = db.areas
coll_cities = db.cities
coll_relations = db.relations

# cities_ids = [2, 60]
cities_ids = [61]
osm_ids = [9450921, 86538, 42602, 3017761, 3148253, 40767]
# osm_ids = [3017761, 3148253, 40767]
for city_idx in cities_ids:
# for osm_id in osm_ids:
    # city = coll_areas.find_one({'osm_id': osm_id})
    city = coll_cities.find_one({'city_id': city_idx})
    shp = shapely.geometry.shape(city['geometry'])
    buffered_polygon = shp.buffer(0.005)
    indexes = h3.polyfill(shapely.geometry.mapping(buffered_polygon), RESOLUTION, geo_json_conformant=False)

    hexes_shp = process_map(check_hex, indexes, max_workers=8)
    hexes_shp = [h for h in hexes_shp if h is not None]
    hexes = [{'hex_id': h['hex_id'], 'geometry': shapely.geometry.mapping(h['geometry'])} for h in tqdm(hexes_shp)]

    # for h in tqdm(indexes):
    #     polygons = h3.h3_set_to_multi_polygon([h], geo_json=False)
    #     outlines = [loop for polygon in polygons for loop in polygon]
    #     polyline = [outline + [outline[0]] for outline in outlines][0]
    #     polygon = shapely.geometry.Polygon(polyline)
    #     if shp.intersects(polygon):
    #         hexes.append({'hex_id': h, 'geometry': shapely.geometry.mapping(polygon)})
    #         hexes_shp.append({'hex_id': h, 'geometry': polygon})

    city_x = generate_x(hexes, NEIGHBORS, NEIGHBORS_EMBEDDING_CLASS, EMBEDDING_CLASS)
    normalized_city_x = normalize_x(city_x)
    print(normalized_city_x.shape)

    # Paris, Ostrava, Munich, Oslo
    city_ids = [38, 41, 42, 45]
    city_ids = [60]
    for inb_ratio in INBALANCE_RATIOS:
        for city_id in city_ids:
            city_y = np.zeros((len(hexes), 2))
            for i in range(ITERATIONS):
                # train_cities_ids = [city['city_id']]
                x, y, hex_ids = generate_xy([city_id], RESOLUTION, inb_ratio, NEIGHBORS, NEIGHBORS_EMBEDDING_CLASS, EMBEDDING_CLASS, i)
                normalized_x = normalize_x(x)
                clf = CLASSIFIER()
                clf.fit(normalized_x, y)
                city_y += clf.predict_proba(normalized_city_x)

            city_y = city_y / (ITERATIONS)
            exists_index = list(clf.classes_).index(1)
            new_list = []
            for h, y in zip(hexes_shp, city_y):
                h['probability'] = y[exists_index]
                new_list.append(h)
            new_df = gpd.GeoDataFrame(new_list)
            del new_df['hex_id']
            new_df.to_file(f"{city_idx}_{city_id}_heatmap_{RESOLUTION}_avg_{ITERATIONS}_cc_rf_inb{inb_ratio}.geojson", driver='GeoJSON')