from .db_definition import db
from .models import *


def init_db():
    db.connect(reuse_if_open = True)
    db.create_tables([Area, Classifier, PredictionResult, TrainLabel, Vector])
