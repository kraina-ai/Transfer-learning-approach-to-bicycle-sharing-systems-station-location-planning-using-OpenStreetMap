from multiprocessing import Pipe, Process

def child_process(func):
    """Makes the function run as a separate process."""
    def wrapper(*args, **kwargs):
        def worker(conn, func, args, kwargs):
            conn.send(func(*args, **kwargs))
            conn.close()
        parent_conn, child_conn = Pipe()
        p = Process(target=worker, args=(child_conn, func, args, kwargs))
        p.start()
        ret = parent_conn.recv()
        p.join()
        return ret
    return wrapper

# @child_process
def get_model(resolution, embedding_cls, input_dim, cities_ids = [], latent_dim = 300, training_size = 50000):
    from .autoencoder_model import Autoencoder
    
    import os

    weights_file = f'weights/{resolution}_{embedding_cls.__name__}_{input_dim}_{latent_dim}_{training_size}_{"_".join(cities_ids)}'
    # print(os.path.exists(weights_file), weights_file)
    if not os.path.exists(weights_file + '.index'):
        from .autoencoder_trainer import train_autoencoder
        print(f'Training model for resolution: {resolution} ({embedding_cls.__name__}, {input_dim}, {latent_dim})')
        train_autoencoder(resolution, embedding_cls, input_dim, cities_ids, latent_dim, training_size)
    
    model = Autoencoder(input_dim, latent_dim)
    model.load_weights(weights_file).expect_partial()

    return model