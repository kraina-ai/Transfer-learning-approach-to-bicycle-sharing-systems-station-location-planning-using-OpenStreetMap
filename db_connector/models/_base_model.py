from peewee import Model
from ..db_definition import db

class BaseModel(Model):
    class Meta:
        database = db