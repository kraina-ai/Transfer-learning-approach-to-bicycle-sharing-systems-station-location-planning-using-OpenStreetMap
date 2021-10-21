from peewee import Field, SqliteDatabase

from json import loads, dumps

class JSONField(Field):
    field_type = 'json'

    def db_value(self, value):
        return dumps(value)  # convert UUID to hex string.

    def python_value(self, value):
        return loads(value) # convert hex string to UUID

db = SqliteDatabase('bike_sharing.db', field_types={'json': 'text'})