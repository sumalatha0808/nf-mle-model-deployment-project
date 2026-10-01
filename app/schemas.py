"""Request and response schemas for the prediction API."""
from pydantic import BaseModel, Field


class Trip(BaseModel):
    PULocationID: int = Field(..., ge=1, le=265, description="Pickup taxi zone ID")
    DOLocationID: int = Field(..., ge=1, le=265, description="Drop-off taxi zone ID")
    trip_distance: float = Field(..., gt=0, le=200, description="Trip distance in miles")

    model_config = {
        "json_schema_extra": {
            "examples": [{"PULocationID": 161, "DOLocationID": 236, "trip_distance": 2.5}]
        }
    }


class Prediction(BaseModel):
    duration_minutes: float
    model_uri: str