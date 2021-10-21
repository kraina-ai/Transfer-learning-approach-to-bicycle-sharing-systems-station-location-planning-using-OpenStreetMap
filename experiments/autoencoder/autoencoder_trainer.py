from .autoencoder_model import Autoencoder
from pymongo import MongoClient
import numpy as np
from tensorflow.keras import losses, callbacks
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def train_autoencoder(resolution, embedding_cls, input_dim, cities_ids = [], latent_dim = 300, training_size = 50000, epochs = 100):
    client = MongoClient('mongodb://localhost:27017/')
    db = client.osmDataDB
    coll_hexes_filtered = db.hexesInCitiesFiltered

    hexes_filter = {}
    if len(cities_ids) > 0:
        hexes_filter['city_id'] = { '$in': cities_ids }
    hexes_filter['has_station'] = True
    hexes_filter['resolution'] = resolution
    all_stations = [hex for hex in coll_hexes_filtered.find(hexes_filter)]
    stations_length = len(all_stations)

    hexes_filter['has_station'] = False
    non_stations_cursor = coll_hexes_filtered.aggregate([
        { '$match': hexes_filter },
        { '$sample': { 'size': training_size - stations_length } }
    ])
    non_stations = [hex for hex in non_stations_cursor]
    y = np.array([1] * stations_length + [0] * len(non_stations))

    relation_embedder = embedding_cls()
    relation_filter = {}
    if len(cities_ids):
        relation_filter['city_id'] = { '$in': cities_ids }
    relation_embedder.fit(relation_filter)
    x = np.array([relation_embedder.transform(h) for h in tqdm(all_stations + non_stations)])
    _train_autoencoder(x, y, resolution, embedding_cls, input_dim, cities_ids, latent_dim, training_size, epochs)

def _train_autoencoder(x, y, resolution, embedding_cls, input_dim, cities_ids = [], latent_dim = 300, training_size = 50000, epochs = 100):
    x_train, x_test = train_test_split(x, test_size=0.2, stratify=y)

    print(x_train.shape, x_test.shape)

    callback = callbacks.EarlyStopping(monitor='loss', patience=5)

    autoencoder = Autoencoder(input_dim, latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(x_train, x_train,
                epochs=epochs,
                shuffle=True,
                callbacks=[callback],
                validation_data=(x_test, x_test))

    autoencoder.save_weights(f'weights/dropout_{epochs}_{resolution}_{embedding_cls.__name__}_{input_dim}_{latent_dim}_{training_size}_{"_".join(cities_ids)}')

if __name__ == "__main__":
    from ..embedding_methods import PerCategoryFilteredShapeAnalyzerEmbedding

    client = MongoClient('mongodb://localhost:27017/')
    db = client.osmDataDB
    coll_hexes_filtered = db.hexesInCitiesFiltered
    hex = coll_hexes_filtered.find_one()
    emb_cls = PerCategoryFilteredShapeAnalyzerEmbedding()
    emb_cls.fit()
    vector = emb_cls.transform(hex)
    input_size = vector.shape[0]
    print(input_size)