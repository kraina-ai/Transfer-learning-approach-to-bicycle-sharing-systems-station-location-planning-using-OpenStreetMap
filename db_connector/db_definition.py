from peewee import Field, PostgresqlDatabase

from json import loads, dumps

class JSONField(Field):
    field_type = 'json'

    def db_value(self, value):
        return dumps(value)  # convert UUID to hex string.

    def python_value(self, value):
        return loads(value) # convert hex string to UUID

db = PostgresqlDatabase('bike_sharing', user='admin', password='admin', host='localhost', port=5432, field_types={'json': 'text'})