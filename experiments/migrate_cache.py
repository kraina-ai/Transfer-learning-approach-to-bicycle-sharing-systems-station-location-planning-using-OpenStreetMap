from pymongo import ASCENDING, GEOSPHERE, MongoClient
from tqdm import tqdm

CHUNK_SIZE = 1000

def yield_rows(cursor, chunk_size):
    """
    Generator to yield chunks from cursor
    :param cursor:
    :param chunk_size:
    :return:
    """
    chunk = []
    for i, row in enumerate(cursor):
        if i % chunk_size == 0 and i > 0:
            yield chunk
            del chunk[:]
        del row['_id']
        chunk.append(row)
    yield chunk

client = MongoClient('mongodb://localhost:27017/')
client_2 = MongoClient('mongodb://localhost:27017/')
db = client.osmDataDB
db_2 = client_2.osmDataDB
coll_embedding_cache = db.hexEmbeddingCache
coll_neighbour_embedding_cache = db.hexNeighbourEmbeddingCache

# coll_embedding_cache
emb_cls_list = coll_embedding_cache.find({}).distinct('embedding_cls')

for emb_cls in emb_cls_list:
    coll_name = f'hexEmbeddingCache{emb_cls}'
    if not coll_name in db.list_collection_names():
        coll_embedding_cache_filtered = db[coll_name]
        cursor = coll_embedding_cache.find({ 'embedding_cls': emb_cls }, batch_size=CHUNK_SIZE)
        chunks = yield_rows(cursor, CHUNK_SIZE)
        for chunk in tqdm(chunks, desc=f'Embedding {emb_cls}', total=cursor.count() // CHUNK_SIZE):
            coll_embedding_cache_filtered.insert_many(chunk)
        coll_embedding_cache_filtered.create_index([("hex_id", ASCENDING)])

# coll_neighbour_embedding_cache
emb_cls_list = coll_neighbour_embedding_cache.find({}).distinct('embedding_cls')

for emb_cls in emb_cls_list:
    coll_name = f'hexNeighbourEmbeddingCache{emb_cls}'
    if not coll_name in db.list_collection_names():
        coll_neighbour_embedding_cache_filtered = db[coll_name]
        cursor = coll_neighbour_embedding_cache.find({ 'embedding_cls': emb_cls }, batch_size=CHUNK_SIZE)
        chunks = yield_rows(cursor, CHUNK_SIZE)
        for chunk in tqdm(chunks, desc=f'Neighbour {emb_cls}', total=cursor.count() // CHUNK_SIZE):
            coll_neighbour_embedding_cache_filtered.insert_many(chunk)
        coll_neighbour_embedding_cache_filtered.create_index([ ("hex_id", ASCENDING), ("ring_no", ASCENDING)])