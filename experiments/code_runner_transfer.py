from __future__ import print_function

import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool, RLock, freeze_support
from random import random
from threading import RLock as TRLock
from time import sleep
import requests

from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map


# %%
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


# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler


# %%
from neighbour_embedding_methods import AverageDiminishingSquqredNeighbourEmbedding, ConcatenateNeighbourEmbedding
from custom_distance_metric import DistanceMetric
from embedding_methods import BaseCountCategoryEmbedding

# %%
INBALANCE_RATIOS = [2.5]
RESOLUTIONS = [11]
NEIGHBORS = {
    11: [5],
}
CLASSIFIERS = [RandomForestClassifier]
NEIGHBORS_EMBEDDING_CLASSES = [AverageDiminishingSquqredNeighbourEmbedding]
EMBEDDING_CLASSES = [BaseCountCategoryEmbedding]
X_PROCESSING = [MinMaxScaler]


# %%
client = MongoClient('mongodb://localhost:27017/')
db = client.osmDataDB
coll_cities = db.cities
coll_hexes_filtered = db.hexesInCitiesFiltered
coll_relations_filtered = db.relationsFiltered

cities = [c for c in coll_cities.find({'accepted': True})]
# cities_from = [c for c in cities if c['city_id'] <= 36]
print([c['city_id'] for c in cities])

# %%
combinations = []
for city_from in cities:
    for city_to in cities:
        for resolution in RESOLUTIONS:
            for inbalance_ratio in INBALANCE_RATIOS:
                for processing in X_PROCESSING:
                    for embedding_cls in EMBEDDING_CLASSES:
                        for neighbours in NEIGHBORS[resolution]:
                            if neighbours == 0:
                                combinations.append({
                                    'city_from': city_from,
                                    'city_to': city_to,
                                    'resolution': resolution,
                                    'inbalance_ratio': inbalance_ratio,
                                    'x_processing': processing,
                                    'embedding_cls': embedding_cls,
                                    'neighbours': neighbours,
                                    'neighbour_embedding_cls': ConcatenateNeighbourEmbedding
                                })
                            else:
                                for neighbour_embedding_cls in NEIGHBORS_EMBEDDING_CLASSES:
                                    combinations.append({
                                        'city_from': city_from,
                                        'city_to': city_to,
                                        'resolution': resolution,
                                        'inbalance_ratio': inbalance_ratio,
                                        'x_processing': processing,
                                        'embedding_cls': embedding_cls,
                                        'neighbours': neighbours,
                                        'neighbour_embedding_cls': neighbour_embedding_cls
                                    })
print(len(combinations))

def normalize_x(x, x_processing):
    new_array = np.zeros(x.shape)
    for i in range(x.shape[1]):
        vector = x[:,i].reshape((-1,1))
        scaler = x_processing()
        new_array[:,i] = scaler.fit_transform(vector).reshape((-1,))
    return new_array

def generate_xy_test(city_id):
    payload = {
        'city_id': city_id,
        'sample': False
    }
    try:
        r = requests.post('http://127.0.0.1:8000/get_data', json=payload)
        data = r.json()
        stations = np.array(data['stations'])
        non_stations = np.array(data['non_stations'])
        x = np.vstack((stations, non_stations))
        y = np.array([1] * stations.shape[0] + [0] * non_stations.shape[0])
        vectors = normalize_x(x, MinMaxScaler)
        return vectors, y
    except Exception as ex:
        print(ex)
        print(payload)
        raise
    # # client = MongoClient('mongodb://localhost:27017/')
    # # coll_hex = client.osmDataDB.hexesInCitiesFiltered
    # # relation_embedder = embedding_cls()
    # # relation_embedder.fit({ 'city_id': city_id })

    # # hexes = [hex for hex in coll_hex.find({ 'city_id': city_id, 'resolution': resolution })]
    # # hex_id_list =  [h['hex_id'] for h in hexes]
    # # y = np.array([int(h['has_station']) for h in hexes])
    # # embedder = neighbour_embedding_cls(relation_embedder)
    # # vectors_list = [embedder.get_embedding(h, neighbours, relation_embedder) for h in tqdm(hexes, desc=f'[City to: {city_id}] Embedding hexes {embedding_cls.__name__} R: {resolution} N: {neighbours}', position=None, lock_args=(False,))]
    # # # vectors = np.stack(vectors_list, axis=0)
    # # # print([v.shape for v in vectors_list])
    # vectors = normalize_x(np.array(vectors_list), MinMaxScaler)
    # # vectors[np.isfinite(vectors) == True] = 0
    # try:
    #     vectors[np.isnan(vectors) == True] = 0
    # except:
    #     print(vectors)

    # # return results_dict
    # return vectors, y, hex_id_list
# %%
def generate_xy(city_id):
    payload = {
        'city_id': city_id,
        'sample': False
    }
    try:
        r = requests.post('http://127.0.0.1:8000/get_data', json=payload)
        data = r.json()
        stations = np.array(data['stations'])
        non_stations = np.array(data['non_stations'])
        x = np.vstack((stations, non_stations))
        y = np.array([1] * stations.shape[0] + [0] * non_stations.shape[0])
        vectors = normalize_x(x, MinMaxScaler)
        return vectors, y
    except Exception as ex:
        print(ex)
        print(payload)
        raise
    # client = MongoClient('mongodb://localhost:27017/')
    # coll_hex = client.osmDataDB.hexesInCitiesFiltered
    # relation_embedder = embedding_cls()
    # relation_embedder.fit({ 'city_id': city_id })

    # max_inbalance_ratio = max(inbalance_ratios)
    # all_stations = [hex for hex in coll_hex.find({ 'city_id': city_id, 'has_station': True, 'resolution': resolution })]
    # stations_length = len(all_stations)
    # non_stations_cursor = coll_hex.aggregate([
    #     { '$match': { 'city_id': city_id, 'has_station': False, 'resolution': resolution } },
    #     { '$sample': { 'size': int(stations_length * max_inbalance_ratio) } }
    # ])
    # non_stations = [hex for hex in non_stations_cursor]
    # hex_id_list =  [h['hex_id'] for h in all_stations + non_stations]
    # y = np.array([1] * stations_length + [0] * len(non_stations))
    # embedder = neighbour_embedding_cls(relation_embedder)
    # vectors_list = [embedder.get_embedding(h, neighbours, relation_embedder) for h in tqdm(all_stations + non_stations, desc=f'[City from: {city_id}] [{id}] Embedding hexes {embedding_cls.__name__} R: {resolution} N: {neighbours} I: {max_inbalance_ratio}', position=None, lock_args=(False,))]
    # # vectors = np.stack(vectors_list, axis=0)
    # # print([v.shape for v in vectors_list])
    # vectors = np.array(vectors_list)
    # # vectors[np.isfinite(vectors) == True] = 0
    # try:
    #     vectors[np.isnan(vectors) == True] = 0
    # except:
    #     print(vectors)

    # try:
    #     results_dict = {}
    #     for imb_r in inbalance_ratios:
    #         for x_processing in x_processings:
    #             key = (imb_r, x_processing)
    #             size = int(stations_length * (imb_r + 1))
    #             # print(imb_r, x_processing, size)
    #             tmp_vectors = vectors[:size, :]
    #             tmp_y = y[:size]
    #             tmp_hex_id_list = hex_id_list[:size]
    #             if x_processing is not None:
    #                 tmp_vectors = normalize_x(tmp_vectors, x_processing)
    #             # print(imb_r, tmp_vectors.shape, len(tmp_y), len(tmp_hex_id_list))
    #             results_dict[key] = (tmp_vectors, tmp_y, tmp_hex_id_list)
    # except Exception as ex:
    #     print(ex)
    #     raise
    # return results_dict
    # # return vectors, y, hex_id_list

def check_if_exists(params):
    city_from = params['city_from']
    city_to = params['city_to']
    resolution = params['resolution']
    inbalance_ratio = params['inbalance_ratio']
    x_processing = params['x_processing']
    embedding_cls = params['embedding_cls']
    neighbours = params['neighbours']
    neighbour_embedding_cls = params['neighbour_embedding_cls'] 

    processing_name = 'None'
    if x_processing is not None:
        processing_name = x_processing.__name__

    client = MongoClient('mongodb://localhost:27017/')
    db = client.osmDataDB
    coll_results = db.experimentsResultsTransferFull
    coll_results.create_index([
        ("city_from", ASCENDING),
        ("city_to", ASCENDING),
        ("resolution", ASCENDING),
        ("inbalance_ratio", ASCENDING),
        ("x_processing", ASCENDING),
        ("embedding_cls", ASCENDING),
        ("neighbours", ASCENDING),
        ("neighbour_embedding_cls", ASCENDING)
    ])

    existing_result = coll_results.find_one({
        "city_from": city_from['city'],
        "city_to": city_to['city'],
        "resolution": resolution,
        "inbalance_ratio": inbalance_ratio,
        "x_processing": processing_name,
        "embedding_cls": embedding_cls.__name__,
        "neighbours": neighbours,
        "neighbour_embedding_cls": neighbour_embedding_cls.__name__
    })

    # print(processing_name)

    return existing_result is not None

# %%
def iterate_combination(t):
    try:
        params, idx = t

        city_from = params['city_from']
        cities_to = params['cities_to']
        resolution = params['resolution']
        inbalance_ratios = params['inbalance_ratios']
        x_processings = params['x_processings']
        embedding_cls = params['embedding_cls']
        neighbours = params['neighbours']
        neighbour_embedding_cls = params['neighbour_embedding_cls'] 

        client = MongoClient('mongodb://localhost:27017/')
        db = client.osmDataDB
        coll_results = db.experimentsResultsTransferFull
        coll_results.create_index([
            ("city_from", ASCENDING),
            ("city_to", ASCENDING),
            ("resolution", ASCENDING),
            ("inbalance_ratio", ASCENDING),
            ("x_processing", ASCENDING),
            ("embedding_cls", ASCENDING),
            ("neighbours", ASCENDING),
            ("neighbour_embedding_cls", ASCENDING)
        ])
        desc = f'[{idx}] {city_from["city"]} Res: {resolution} EmbCls: {embedding_cls.__name__} NeighEmbCls: {neighbour_embedding_cls.__name__} Neigh: {neighbours}'
        # if existing_result is None:
        results = {}
        ITERATIONS = 100
        with tqdm(total=ITERATIONS * len(CLASSIFIERS) * len(inbalance_ratios) * len(x_processings) * len(cities_to), desc=desc, position=None, lock_args=(False,)) as pbar:
            for city_to in cities_to:
                custom_metric = DistanceMetric()
                custom_metric.fit({'city_id': city_to['city_id'], 'resolution': resolution})
                test_x, test_y = generate_xy_test(city_to['city_id'])
                print(city_to['city'], test_x.shape, test_y.shape)
                results = []
                for iteration in range(ITERATIONS):
                    train_x, train_y = generate_xy(city_from['city_id'])
                    X_train = np.array(train_x)
                    Y_train = np.array(train_y)
                    X_test = np.array(test_x)
                    Y_test = np.array(test_y)

                    for clf_cls in CLASSIFIERS:
                        clf = clf_cls()
                        try:
                            clf.fit(X_train, Y_train)
                        except:
                            print(X_train)
                            print(X_train.shape)
                            print(np.any(np.isnan(X_train)))
                            print(np.all(np.isfinite(X_train)))
                            raise

                        try:
                            y_pred = clf.predict(X_test)
                        except:
                            print(X_test)
                            print(X_test.shape)
                            print(np.any(np.isnan(X_test)))
                            print(np.all(np.isfinite(X_test)))
                            raise

                        acc = accuracy_score(y_pred=y_pred, y_true=Y_test)
                        f1 = f1_score(y_pred=y_pred, y_true=Y_test)
                        prec = precision_score(y_pred=y_pred, y_true=Y_test)
                        rec = recall_score(y_pred=y_pred, y_true=Y_test)
                        bal_acc = balanced_accuracy_score(y_pred=y_pred, y_true=Y_test)
                        # custom_metric_value = custom_metric.calculate(Y_test, y_pred, test_hex_ids)
                        results.append({
                            'classfier_cls': clf_cls.__name__,
                            'iteration': iteration + 1,
                            'dataset_type': 'validation',
                            'accuracy': acc,
                            'f1_score': f1,
                            'custom_metric': -1,
                            'precision': prec,
                            'recall': rec,
                            'balanced_accuracy': bal_acc
                        })
                        pbar.update(1)

                coll_results.insert_one({
                    'city_from': city_from['city'],
                    'city_to': city_to['city'],
                    'resolution': resolution,
                    'inbalance_ratio': 2.5,
                    'x_processing': 'MinMaxScaler',
                    'embedding_cls': embedding_cls.__name__,
                    'neighbours': neighbours,
                    'neighbour_embedding_cls': neighbour_embedding_cls.__name__,
                    'results': results
                })
        
        # for (imb_r, processing_name, city_to_name), results_list in results.items():
        #     coll_results.insert_one({
        #         'city_from': city_from['city'],
        #         'city_to': city_to_name,
        #         'resolution': resolution,
        #         'inbalance_ratio': imb_r,
        #         'x_processing': processing_name,
        #         'embedding_cls': embedding_cls.__name__,
        #         'neighbours': neighbours,
        #         'neighbour_embedding_cls': neighbour_embedding_cls.__name__,
        #         'results': results_list
        #     })
        tqdm.write(f"Finished {desc}")
    except Exception as ex:
        print(ex)
        raise


# %%
# skip = 77
# for idx, combination in enumerate(combinations):
#     if idx < skip:
#         continue
#     iterate_combination(combination, idx)


# %%
from functools import partial
from itertools import chain

from tqdm.contrib.concurrent import process_map
pairs = []

filtered_combinations = []
counter = 1
# skip = 88
for combination in tqdm(combinations, desc="Checking existing results", total=len(combinations)):
    if not check_if_exists(combination):
        filtered_combinations.append(combination)
        counter += 1

print(len(filtered_combinations))
# process_map(iterate_combination, pairs, max_workers=6)

grouped_combinations = {}
def group_by_processing_and_balance_ratio():
    for params in filtered_combinations:
        city_from = params['city_from']
        city_to = params['city_to']
        resolution = params['resolution']
        embedding_cls = params['embedding_cls']
        neighbours = params['neighbours']
        neighbour_embedding_cls = params['neighbour_embedding_cls']
        
        key = (city_from['city'], city_to['city'], resolution, embedding_cls, neighbours, neighbour_embedding_cls)

        inbalance_ratio = params['inbalance_ratio']
        x_processing = params['x_processing']

        if not key in grouped_combinations:
            grouped_combinations[key] = {
                'city_from': city_from,
                'cities_to': [city_to],
                'resolution': resolution,
                'embedding_cls': embedding_cls,
                'neighbours': neighbours,
                'neighbour_embedding_cls': neighbour_embedding_cls,
                'inbalance_ratios': set(),
                'x_processings': set(),
            }
        grouped_combinations[key]['inbalance_ratios'].add(inbalance_ratio)
        grouped_combinations[key]['x_processings'].add(x_processing)

group_by_processing_and_balance_ratio()
grouped_combinations_list = [(params, i + 1) for i, params in enumerate(grouped_combinations.values())]
print(len(grouped_combinations_list))
# print(len(grouped_combinations_list))
# print(grouped_combinations_list[0])

# %%

tqdm.set_lock(RLock())
p = Pool(processes=10, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
# p = Pool(processes=4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
# p = Pool(processes=1, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
p.map(partial(iterate_combination), grouped_combinations_list)

# for pair in grouped_combinations_list:
#     iterate_combination(pair)

# PY2 = sys.version_info[:1] <= (2,)
# tqdm.set_lock(TRLock())
# pool_args = {}
# if not PY2:
#     pool_args.update(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
# with ThreadPoolExecutor(**pool_args) as p:
#     p.map(partial(iterate_combination), pairs)


