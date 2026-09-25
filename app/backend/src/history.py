from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database import get_database


COLLECTION_NAME = "prediction_history"


def save_prediction_history(
    request_id: str,
    features: Dict[str, Any],
    prediction: int,
    probability: Optional[float] = None,
    model_version: Optional[str] = None,
) -> str:
    database = get_database()
    collection = database[COLLECTION_NAME]

    document = {
        "request_id": request_id,
        "features": features,
        "prediction": prediction,
        "probability": probability,
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc),
    }

    result = collection.insert_one(document)

    return str(result.inserted_id)


def get_prediction_history(
    limit: int = 20,
) -> List[Dict[str, Any]]:
    database = get_database()
    collection = database[COLLECTION_NAME]

    cursor = (
        collection
        .find()
        .sort("created_at", -1)
        .limit(limit)
    )

    history = []

    for document in cursor:
        document["_id"] = str(document["_id"])
        history.append(document)

    return history