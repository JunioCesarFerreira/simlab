import json
from bson import ObjectId
from datetime import datetime

# Função auxiliar para converter tipos não serializáveis (como ObjectId)
def json_encoder(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Tipo não serializável: {type(obj)}")