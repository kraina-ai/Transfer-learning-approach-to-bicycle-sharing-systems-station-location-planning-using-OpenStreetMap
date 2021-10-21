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

from functools import partial
from multiprocessing import Pool, RLock
from threading import RLock as TRLock

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

RESOLUTION = 9
NEIGHBORS = 0
# NEIGHBORS = 0
NEIGHBORS_EMBEDDING_CLASS = AverageDiminishingSquqredNeighbourEmbedding
EMBEDDING_CLASS = BaseCountCategoryEmbedding

# client = MongoClient('mongodb://localhost:27017/')
# db = client.osmDataDB
# coll_cities = db.cities
# coll_relations = db.relations

# city = coll_cities.find_one({'city': 'Wrocław'})
# shp = shapely.geometry.shape(city['geometry'])
# buffered_polygon = shp.buffer(0.005)
# indexes = h3.polyfill(shapely.geometry.mapping(buffered_polygon), RESOLUTION, geo_json_conformant=False)

# hexes = []
# hexes_shp = []
# for h in tqdm(indexes):
#     polygons = h3.h3_set_to_multi_polygon([h], geo_json=False)
#     outlines = [loop for polygon in polygons for loop in polygon]
#     polyline = [outline + [outline[0]] for outline in outlines][0]
#     polygon = shapely.geometry.Polygon(polyline)
#     if shp.intersects(polygon):
#         hexes.append({'hex_id': h, 'geometry': shapely.geometry.mapping(polygon)})
#         hexes_shp.append({'hex_id': h, 'geometry': polygon})

from functools import partial
from tqdm.contrib.concurrent import process_map 


def _embed(h_tuple):
    hexes, i = h_tuple
    client = MongoClient('mongodb://localhost:27017/')
    coll_hex = client.osmDataDB.hexesInCitiesFiltered
    hexes = [coll_hex.find_one({ 'hex_id': h }) for h in tqdm(hexes, desc=f'[{i}] Finding', position=None, lock_args=(False,))]
    relation_embedder = EMBEDDING_CLASS()
    relation_embedder.fit({}, filtered = True)
    embedder = NEIGHBORS_EMBEDDING_CLASS(relation_embedder)
    for h in tqdm(hexes, desc=f'[{i}] Embedding', position=None, lock_args=(False,)):
        embedder.get_embedding(h, NEIGHBORS, relation_embedder, check_if_exists=True)
    return 0

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield (lst[i:i + n], i / n)

# def generate_x(hexes, neighbours, neighbour_embedding_cls, embedding_cls, id=None):
#     client = MongoClient('mongodb://localhost:27017/')
#     coll_hex = client.osmDataDB.hexesInCitiesFiltered
#     relation_embedder = embedding_cls()
#     relation_embedder.fit({ 'city_id': city_id })

#     max_inbalance_ratio = max(inbalance_ratios)
#     all_stations = [hex for hex in coll_hex.find({ 'city_id': city_id, 'resolution': resolution })]
#     stations_length = len(all_stations)
#     non_stations_cursor = coll_hex.aggregate([
#         { '$match': { 'city_id': city_id, 'has_station': False, 'resolution': resolution } },
#         { '$sample': { 'size': int(stations_length * max_inbalance_ratio) } }
#     ])
#     non_stations = [hex for hex in non_stations_cursor]
#     hex_id_list =  [h['hex_id'] for h in all_stations + non_stations]
#     y = np.array([1] * stations_length + [0] * len(non_stations))
#     embedder = neighbour_embedding_cls(relation_embedder)
#     vectors_list = [embedder.get_embedding(h, neighbours, relation_embedder) for h in tqdm(all_stations + non_stations, desc=f'[{id}] [{city_id}] Embedding hexes {embedding_cls.__name__} R: {resolution} N: {neighbours} I: {max_inbalance_ratio}', position=None, lock_args=(False,))]
    
#     relation_embedder = embedding_cls()
#     relation_embedder.fit({}, filtered = False)
#     embedder = neighbour_embedding_cls(relation_embedder)
#     # _embed = partial(embedder.get_embedding, no_rings = neighbours, embedding_cls = relation_embedder, check_if_exists = False)
#     vectors_chunk_lists = process_map(_embed, list(chunks(hexes, 1000)), max_workers=8)
#     # vectors_list = [embedder.get_embedding(h, neighbours, relation_embedder, check_if_exists=False) for h in tqdm(hexes, desc=f'[{id}] Embedding hexes {embedding_cls.__name__}')]
#     vectors = np.array([v for v_list in vectors_chunk_lists for v in v_list])
#     try:
#         vectors[np.isnan(vectors) == True] = 0
#     except:
#         print(vectors)
#     return vectors

client = MongoClient('mongodb://localhost:27017/')
coll_hex = client.osmDataDB.hexesInCitiesFiltered
# coll_hex_cache = client.osmDataDB.hexNeighbourEmbeddingCacheBaseCountCategoryEmbeddingFull
# coll_hex_cache = client.osmDataDB.hexNeighbourEmbeddingCacheBaseShapeAnalyzerEmbedding
# cursor = coll_hex.aggregate([
#     {
#         '$match': {
#             'resolution': 11
#         }
#     }, {
#         '$lookup': {
#             'from': 'hexNeighbourEmbeddingCacheBaseShapeAnalyzerEmbedding', 
#             'localField': 'hex_id', 
#             'foreignField': 'hex_id', 
#             'as': 'cache'
#         }
#     }, {
#         '$match': {
#             'cache': {
#                 '$exists': True, 
#                 '$type': 'array', 
#                 '$size': 0
#             }
#         }
#     }, {
#         '$project': {
#             'hex_id': 1, 
#             '_id': 0
#         }
#     }
# ], allowDiskUse = True)
# hexes = [h['hex_id'] for h in tqdm(coll_hex.find({ 'resolution': RESOLUTION, "city_id": { "$gte": 37 } }, { "hex_id": 1 }))]
hexes_raw = set([h['hex_id'] for h in tqdm(coll_hex.find({ 'resolution': RESOLUTION }, { "hex_id": 1 }))])
# hexes_cached = set([h['hex_id'] for h in tqdm(coll_hex_cache.find({}, { "hex_id": 1 }))])
# hexes = list(hexes_raw - hexes_cached)
hexes = list(hexes_raw)
del hexes_raw
# del hexes_cached
chunks = list(chunks(hexes, 10000))
print(len(chunks))
del hexes

tqdm.set_lock(RLock())
p = Pool(processes=8, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
p.map(partial(_embed), chunks)
