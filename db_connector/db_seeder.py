from .db_definition import db
from .models import Area, TrainVector, Vector


def init_db():
    db.connect(reuse_if_open = True)
    db.create_tables([Area, TrainVector, Vector])
