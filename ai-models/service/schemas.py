from pydantic import BaseModel, ConfigDict, Field


class WaterQualityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ph: float = Field(
        ...,
        ge=0.0,
        le=13.999999999999998,
    )

    Hardness: float = Field(
        ...,
        ge=47.432,
        le=323.124,
    )

    Solids: float = Field(
        ...,
        ge=320.942611274359,
        le=61227.19600771213,
    )

    Chloramines: float = Field(
        ...,
        ge=0.3520000000000003,
        le=13.127000000000002,
    )

    Sulfate: float = Field(
        ...,
        ge=129.00000000000003,
        le=481.0306423059972,
    )

    Conductivity: float = Field(
        ...,
        ge=181.483753985146,
        le=753.3426195583046,
    )

    Organic_carbon: float = Field(
        ...,
        ge=2.1999999999999886,
        le=28.30000000000001,
    )

    Trihalomethanes: float = Field(
        ...,
        ge=0.7379999999999995,
        le=124.0,
    )

    Turbidity: float = Field(
        ...,
        ge=1.45,
        le=6.739,
    )