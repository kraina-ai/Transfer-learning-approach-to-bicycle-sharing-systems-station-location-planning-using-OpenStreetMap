from embedding_methods import PerCategoryFilteredShapeAnalyzerEmbedding
from autoencoder.autoencoder_trainer import train_autoencoder, _train_autoencoder
from pymongo import MongoClient
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map  # or thread_map

client = MongoClient('mongodb://localhost:27017/')
db = client.osmDataDB
coll_hexes_filtered = db.hexesInCitiesFiltered
hex = coll_hexes_filtered.find_one()
emb_clf = PerCategoryFilteredShapeAnalyzerEmbedding()
emb_clf.fit({})
vector = emb_clf.transform(hex)
input_size = vector.shape[0]
print(input_size)
training_size = {
    9: 100000,
    10: 100000,
    11: 100000
}
epochs = 100
# training_size = 10000

def _embed(hex):
    global emb_clf
    return emb_clf.transform(hex)

# for res in [9, 10, 11]:
for res in [9, 10, 11]:
    ts = training_size[res]
    client = MongoClient('mongodb://localhost:27017/')
    db = client.osmDataDB
    coll_hexes_filtered = db.hexesInCitiesFiltered

    hexes_filter = {}
    hexes_filter['has_station'] = True
    hexes_filter['resolution'] = res
    all_stations = [hex for hex in coll_hexes_filtered.find(hexes_filter)]
    stations_length = len(all_stations)

    hexes_filter['has_station'] = False
    non_stations_cursor = coll_hexes_filtered.aggregate([
        { '$match': hexes_filter },
        { '$sample': { 'size': ts - stations_length } }
    ], allowDiskUse = True)
    non_stations = [hex for hex in non_stations_cursor]
    y = np.array([1] * stations_length + [0] * len(non_stations))
    # x = np.array([emb_clf.transform(h) for h in tqdm(all_stations + non_stations)])
    x = np.array(process_map(_embed, all_stations + non_stations, max_workers=8))
    # x = np.array([emb_clf.transform for h in tqdm(all_stations + non_stations)])

    # print(x)

    # for final_dim in [32, 50, 64, 100, 128, 200, 256, 300, 500]:
    for final_dim in [20]:
        # resolution, embedding_cls, input_dim, cities_ids = [], latent_dim = 300, training_size = 50000
        _train_autoencoder(x, y, res, PerCategoryFilteredShapeAnalyzerEmbedding, input_size, latent_dim = final_dim, training_size = ts, epochs = epochs)