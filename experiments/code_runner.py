from __future__ import print_function

import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool, RLock, freeze_support
from random import random
from threading import RLock as TRLock
from time import sleep

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from balanced_rf import BalancedRandomForestClassifier, BalancedSubsampleRandomForestClassifier
from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# %%
from neighbour_embedding_methods import AverageDiminishingNeighbourEmbedding, AverageDiminishingSquqredNeighbourEmbedding, AverageNeighbourEmbedding, ConcatenateNeighbourEmbedding
from custom_distance_metric import DistanceMetric
from embedding_methods import BaseCountCategoryEmbedding, BaseShapeAnalyzerEmbedding, PerCategoryShapeAnalyzerEmbedding, PerCategoryShapeAnalyzerEmbedding300
from embedding_methods import PerCategoryFilteredShapeAnalyzerEmbedding20, PerCategoryFilteredShapeAnalyzerEmbedding32, \
                            PerCategoryFilteredShapeAnalyzerEmbedding50, PerCategoryFilteredShapeAnalyzerEmbedding64, \
                            PerCategoryFilteredShapeAnalyzerEmbedding100, PerCategoryFilteredShapeAnalyzerEmbedding128, \
                            PerCategoryFilteredShapeAnalyzerEmbedding200, PerCategoryFilteredShapeAnalyzerEmbedding256, \
                            PerCategoryFilteredShapeAnalyzerEmbedding300, PerCategoryFilteredShapeAnalyzerEmbedding500

# %%
# INBALANCE_RATIOS = [1, 2, 3, 5] 
# INBALANCE_RATIOS = list(np.arange(1, 5.5, 0.5)) 
INBALANCE_RATIOS = [2.5]
# print(INBALANCE_RATIOS)
RESOLUTIONS = [9, 10, 11]
# RESOLUTIONS = [11]
# NEIGHBORS = {
#     9: [0,1,2,3],
#     10: [0,2,4,6,8,10],
#     11: [0,5,10,15,20,25]
# }
NEIGHBORS = {
    9: list(range(4)),
    10: list(range(11)),
    11: list(range(26))
}
print(NEIGHBORS)
# CLASSIFIERS = [KNeighborsClassifier, SVC, RandomForestClassifier, AdaBoostClassifier]
CLASSIFIERS = [RandomForestClassifier]
# CLASSIFIERS = [BalancedRandomForestClassifier, BalancedSubsampleRandomForestClassifier]
# NEIGHBORS_EMBEDDING_CLASSES = [ConcatenateNeighbourEmbedding, AverageNeighbourEmbedding, AverageDiminishingNeighbourEmbedding, AverageDiminishingSquqredNeighbourEmbedding]
NEIGHBORS_EMBEDDING_CLASSES = [AverageDiminishingSquqredNeighbourEmbedding]
# NEIGHBORS_EMBEDDING_CLASSES = [ConcatenateNeighbourEmbedding, AverageDiminishingSquqredNeighbourEmbedding]
# EMBEDDING_CLASSES = [BaseShapeAnalyzerEmbedding]
# EMBEDDING_CLASSES = [BaseCountCategoryEmbedding, BaseShapeAnalyzerEmbedding]
EMBEDDING_CLASSES = [BaseCountCategoryEmbedding]
# EMBEDDING_CLASSES = [BaseCountCategoryEmbedding, BaseShapeAnalyzerEmbedding, PerCategoryShapeAnalyzerEmbedding300, PerCategoryFilteredShapeAnalyzerEmbedding300]
# EMBEDDING_CLASSES = [PerCategoryFilteredShapeAnalyzerEmbedding20, PerCategoryFilteredShapeAnalyzerEmbedding32, \
#     PerCategoryFilteredShapeAnalyzerEmbedding50, PerCategoryFilteredShapeAnalyzerEmbedding64, \
#     PerCategoryFilteredShapeAnalyzerEmbedding100, PerCategoryFilteredShapeAnalyzerEmbedding128, \
#     PerCategoryFilteredShapeAnalyzerEmbedding200, PerCategoryFilteredShapeAnalyzerEmbedding256, \
#     PerCategoryFilteredShapeAnalyzerEmbedding300, PerCategoryFilteredShapeAnalyzerEmbedding500
# ]
# EMBEDDING_CLASSES = [PerCategoryFilteredShapeAnalyzerEmbedding300]
# X_PROCESSING = [None, MinMaxScaler, StandardScaler]
X_PROCESSING = [MinMaxScaler]


# %%
client = MongoClient('mongodb://localhost:27017/')
db = client.osmDataDB
coll_cities = db.cities
coll_hexes_filtered = db.hexesInCitiesFiltered
coll_relations_filtered = db.relationsFiltered


arguments = len(sys.argv) - 1
position = 1
city_ids = []
while (arguments >= position):
    city_ids.append(int(sys.argv[position]))
    position = position + 1
    

print(city_ids)

# %%
if len(city_ids) == 0:
    city_ids = [c['city_id'] for c in coll_cities.find(
        {
            'accepted': True,
            # 'city_id': { '$in': [58, 59] }
            # 'city_id': { '$in': [58, 59, 46, 3, 30, 45, 47] }
            # 'city_id': { '$in': [1] }
        },
        {'_id': 0, 'city_id': 1}
    )]
# cities

cities = []
# for city_id in [58]:
# for city_id in [59, 46, 3, 30, 45, 47]:
for city_id in city_ids:
    cities.append(coll_cities.find_one({'city_id': city_id}))


# %%
combinations = []
for city in cities:
    for resolution in RESOLUTIONS:
        for inbalance_ratio in INBALANCE_RATIOS:
            for processing in X_PROCESSING:
                for embedding_cls in EMBEDDING_CLASSES:
                    for neighbours in NEIGHBORS[resolution]:
                        if neighbours == 0:
                            combinations.append({
                                'city': city,
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
                                    'city': city,
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

# %%
def generate_xy(city_id, resolution, inbalance_ratios, x_processings, neighbours, neighbour_embedding_cls, embedding_cls, id=None):
    client = MongoClient('mongodb://localhost:27017/')
    coll_hex = client.osmDataDB.hexesInCitiesFiltered
    relation_embedder = embedding_cls()
    relation_embedder.fit({ 'city_id': city_id })

    max_inbalance_ratio = max(inbalance_ratios)
    all_stations = [hex for hex in coll_hex.find({ 'city_id': city_id, 'has_station': True, 'resolution': resolution })]
    stations_length = len(all_stations)
    non_stations_cursor = coll_hex.aggregate([
        { '$match': { 'city_id': city_id, 'has_station': False, 'resolution': resolution } },
        { '$sample': { 'size': int(stations_length * max_inbalance_ratio) } }
    ])
    non_stations = [hex for hex in non_stations_cursor]
    hex_id_list =  [h['hex_id'] for h in all_stations + non_stations]
    y = np.array([1] * stations_length + [0] * len(non_stations))
    embedder = neighbour_embedding_cls(relation_embedder)
    vectors_list = [embedder.get_embedding(h, neighbours, relation_embedder) for h in tqdm(all_stations + non_stations, desc=f'[{id}] Embedding hexes {embedding_cls.__name__} R: {resolution} N: {neighbours} I: {max_inbalance_ratio}', position=None, lock_args=(False,))]
    # vectors = np.stack(vectors_list, axis=0)
    # print([v.shape for v in vectors_list])
    vectors = np.array(vectors_list)
    # vectors[np.isfinite(vectors) == True] = 0
    try:
        vectors[np.isnan(vectors) == True] = 0
    except:
        print(vectors)

    try:
        results_dict = {}
        for imb_r in inbalance_ratios:
            for x_processing in x_processings:
                key = (imb_r, x_processing)
                size = int(stations_length * (imb_r + 1))
                # print(imb_r, x_processing, size)
                tmp_vectors = vectors[:size, :]
                tmp_y = y[:size]
                tmp_hex_id_list = hex_id_list[:size]
                if x_processing is not None:
                    tmp_vectors = normalize_x(tmp_vectors, x_processing)
                # print(imb_r, tmp_vectors.shape, len(tmp_y), len(tmp_hex_id_list))
                results_dict[key] = (tmp_vectors, tmp_y, tmp_hex_id_list)
    except Exception as ex:
        print(ex)
        raise
    return results_dict
    # return vectors, y, hex_id_list

def check_if_exists(params):
    city = params['city']
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
    coll_results = db.experimentsResults
    coll_results.create_index([
        ("city", ASCENDING),
        ("resolution", ASCENDING),
        ("inbalance_ratio", ASCENDING),
        ("x_processing", ASCENDING),
        ("embedding_cls", ASCENDING),
        ("neighbours", ASCENDING),
        ("neighbour_embedding_cls", ASCENDING)
    ])

    existing_result = coll_results.find_one({
        "city": city['city'],
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

        city = params['city']
        resolution = params['resolution']
        inbalance_ratios = params['inbalance_ratios']
        x_processings = params['x_processings']
        embedding_cls = params['embedding_cls']
        neighbours = params['neighbours']
        neighbour_embedding_cls = params['neighbour_embedding_cls'] 
        best_clf_dict = {}

        # processing_name = 'None'
        # if x_processing is not None:
        #     processing_name = x_processing.__name__

        client = MongoClient('mongodb://localhost:27017/')
        db = client.osmDataDB
        coll_results = db.experimentsResults
        coll_results.create_index([
            ("city", ASCENDING),
            ("resolution", ASCENDING),
            ("inbalance_ratio", ASCENDING),
            ("x_processing", ASCENDING),
            ("embedding_cls", ASCENDING),
            ("neighbours", ASCENDING),
            ("neighbour_embedding_cls", ASCENDING)
        ])
        desc = f'[{idx}] {city["city"]} Res: {resolution} EmbCls: {embedding_cls.__name__} NeighEmbCls: {neighbour_embedding_cls.__name__} Neigh: {neighbours}'

        # if existing_result is None:
        custom_metric = DistanceMetric()
        custom_metric.fit({'city_id': city['city_id'], 'resolution': resolution})
        results = {}
        with tqdm(total=10 * 10 * len(CLASSIFIERS) * len(inbalance_ratios) * len(x_processings), desc=desc, position=None, lock_args=(False,)) as pbar:
            for iteration in range(10):
                tuples_dict = generate_xy(city['city_id'], resolution, inbalance_ratios, x_processings, neighbours, neighbour_embedding_cls, embedding_cls, idx)
                # x, y, hex_ids = generate_xy(city['city_id'], resolution, inbalance_ratios, x_processings, neighbours, neighbour_embedding_cls, embedding_cls, idx)

                for (imb_r, x_processing), (x, y, hex_ids) in tuples_dict.items():
                    processing_name = 'None'
                    if x_processing is not None:
                        processing_name = x_processing.__name__
                    key = (imb_r, processing_name)
                    if not key in results:
                        results[key] = []

                    y_hex_zip = list(zip(y, hex_ids))

                    X_tmp, X_validation, Y_hex_id_tmp, Y_hex_id_validation = train_test_split(x, y_hex_zip, test_size=0.2, stratify=y)

                    X_validation = np.array(X_validation)
                    Y_validation, hex_ids_validation = zip(*Y_hex_id_validation)
                    Y_validation = np.array(list(Y_validation))
                    hex_ids_validation = list(hex_ids_validation)

                    y_tmp = [t[0] for t in Y_hex_id_tmp]

                    clf_values = {}

                    for _ in range(10):
                        X_train, X_test, Y_hex_id_train, Y_hex_id_test = train_test_split(X_tmp, Y_hex_id_tmp, test_size=0.25, stratify=y_tmp)
                        X_train = np.array(X_train)
                        Y_train, hex_ids_train = zip(*Y_hex_id_train)
                        Y_train = np.array(list(Y_train))
                        hex_ids_train = list(hex_ids_train)
                        X_test = np.array(X_test)
                        Y_test, hex_ids_test = zip(*Y_hex_id_test)
                        Y_test = np.array(list(Y_test))
                        hex_ids_test = list(hex_ids_test)
                        best_clf_dict = {}

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
                            custom_metric_value = custom_metric.calculate(Y_test, y_pred, hex_ids_test)
                            if not clf_cls.__name__ in clf_values:
                                clf_values[clf_cls.__name__] = {
                                    'accuracy': [],
                                    'f1': [],
                                    'custom_metric': [],
                                    'precision': [],
                                    'recall': [],
                                    'balanced_accuracy': [],
                                }
                            clf_values[clf_cls.__name__]['accuracy'].append(acc)
                            clf_values[clf_cls.__name__]['f1'].append(f1)
                            clf_values[clf_cls.__name__]['custom_metric'].append(custom_metric_value)
                            clf_values[clf_cls.__name__]['precision'].append(prec)
                            clf_values[clf_cls.__name__]['recall'].append(rec)
                            clf_values[clf_cls.__name__]['balanced_accuracy'].append(bal_acc)
                            if not clf_cls.__name__ in best_clf_dict or f1 > best_clf_dict[clf_cls.__name__][0]:
                                best_clf_dict[clf_cls.__name__] = (f1, clf)
                            pbar.update(1)
            
                    for k, v in clf_values.items():
                        results[key].append({
                            # 'city': city['city'],
                            # 'resolution': resolution,
                            # 'inbalance_ratio': inbalance_ratio,
                            # 'embedding_cls': embedding_cls.__name__,
                            # 'neighbours': neighbours,
                            # 'neighbour_embedding_cls': neighbour_embedding_cls.__name__,
                            'classfier_cls': k,
                            'iteration': iteration + 1,
                            'dataset_type': 'test',
                            'accuracy': np.average(v['accuracy']),
                            'f1_score': np.average(v['f1']),
                            'custom_metric': np.average(v['custom_metric']),
                            'precision': np.average(v['precision']),
                            'recall': np.average(v['recall']),
                            'balanced_accuracy': np.average(v['balanced_accuracy'])
                        })
                    
                    for k, v in best_clf_dict.items():
                        y_pred = v[1].predict(X_validation)
                        acc = accuracy_score(y_pred=y_pred, y_true=Y_validation)
                        f1 = f1_score(y_pred=y_pred, y_true=Y_validation)
                        prec = precision_score(y_pred=y_pred, y_true=Y_validation)
                        rec = recall_score(y_pred=y_pred, y_true=Y_validation)
                        bal_acc = balanced_accuracy_score(y_pred=y_pred, y_true=Y_validation)
                        custom_metric_value = custom_metric.calculate(Y_validation, y_pred, hex_ids_validation)
                        results[key].append({
                            # 'city': city['city'],
                            # 'resolution': resolution,
                            # 'inbalance_ratio': inbalance_ratio,
                            # 'embedding_cls': embedding_cls.__name__,
                            # 'neighbours': neighbours,
                            # 'neighbour_embedding_cls': neighbour_embedding_cls.__name__,
                            'classfier_cls': k,
                            'iteration': iteration + 1,
                            'dataset_type': 'validation',
                            'accuracy': acc,
                            'f1_score': f1,
                            'custom_metric': custom_metric_value,
                            'precision': prec,
                            'recall': rec,
                            'balanced_accuracy': bal_acc
                        })
        
        for (imb_r, processing_name), results_list in results.items():
            coll_results.insert_one({
                'city': city['city'],
                'resolution': resolution,
                'inbalance_ratio': imb_r,
                'x_processing': processing_name,
                'embedding_cls': embedding_cls.__name__,
                'neighbours': neighbours,
                'neighbour_embedding_cls': neighbour_embedding_cls.__name__,
                'results': results_list
            })
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
        city = params['city']
        resolution = params['resolution']
        embedding_cls = params['embedding_cls']
        neighbours = params['neighbours']
        neighbour_embedding_cls = params['neighbour_embedding_cls']
        
        key = (city['city'], resolution, embedding_cls, neighbours, neighbour_embedding_cls)

        inbalance_ratio = params['inbalance_ratio']
        x_processing = params['x_processing']

        if not key in grouped_combinations:
            grouped_combinations[key] = {
                'city': city,
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

# %%

tqdm.set_lock(RLock())
p = Pool(processes=16, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
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


