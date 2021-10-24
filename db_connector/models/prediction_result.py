from peewee import CharField, CompositeKey, DecimalField, ForeignKeyField

from ._base_model import BaseModel
from .area import Area
from .classifier import Classifier


class PredictionResult(BaseModel):
    hex_id = CharField(max_length = 15, index = True)
    classifier = ForeignKeyField(Classifier)
    area = ForeignKeyField(Area)
    value = DecimalField()

    class Meta:
        primary_key = CompositeKey('hex_id', 'classifier')
    