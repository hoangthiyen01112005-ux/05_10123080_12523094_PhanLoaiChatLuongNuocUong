import os

from pymongo import MongoClient
from pymongo.database import Database


MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017",
)

MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "water_quality",
)


_client = None


def get_mongo_client() -> MongoClient:
    global _client

    if _client is None:
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
        )

    return _client


def get_database() -> Database:
    client = get_mongo_client()
    return client[MONGODB_DATABASE]


def check_database_connection() -> bool:
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        return True

    except Exception:
        return False