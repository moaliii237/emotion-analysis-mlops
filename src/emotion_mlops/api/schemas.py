"""
Pydantic schemas for the API.

This file only defines JSON shapes, no business logic.
FastAPI uses these automatically for:
  - Validating incoming JSON (request validation)
  - Serializing outgoing JSON (response formatting)
  - Generating the /docs Swagger UI
"""

from typing import Optional

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """Shape of the JSON body for POST /predict/text."""

    text: str = Field(
        ...,
        min_length=1,
        description="The sentence to classify.",
        examples=["Listen to me right now!"],
    )


class PredictionResponse(BaseModel):
    """Shape of the JSON response from POST /predict/text."""

    text: str
    emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class FullPredictionResponse(BaseModel):
    """Shape of the JSON response from POST /predict/text/full, used by the UI."""

    text: str
    top_emotion: str
    top_confidence: float = Field(..., ge=0.0, le=1.0)
    all_emotions: dict[str, float] = Field(
        ...,
        description="Probability for each of the 7 emotion labels.",
    )
    model_version: str


class HealthResponse(BaseModel):
    """Shape of the JSON response from GET /health."""

    status: str
    model_loaded: bool
    model_version: str | None = None


# ===========================================================================
# Feedback (Card #1486, continuous retraining with user feedback)
# ===========================================================================


class FeedbackRequest(BaseModel):
    """User correction (or confirmation) of a model prediction."""

    sentence: str = Field(..., min_length=1, description="The original input sentence.")
    predicted_label: str = Field(..., description="What the model predicted.")
    predicted_conf: float = Field(..., ge=0.0, le=1.0, description="Model confidence.")
    is_correct: bool = Field(..., description="True if user agreed with model.")
    user_label: Optional[str] = Field(
        None,
        description="User's corrected label (required if is_correct=False).",
    )
    model_version: str = Field(..., description="Which model produced the prediction.")
    source: Optional[str] = Field(
        None,
        description="Where the prediction came from: 'audio', 'transcript', or 'text'.",
    )
    file_id: Optional[str] = Field(None, description="Source file identifier.")


class FeedbackResponse(BaseModel):
    """Acknowledgement returned after recording feedback."""

    id: int = Field(..., description="The new row's ID in the feedback DB.")
    message: str = "Feedback recorded."


class FeedbackStatsPerModel(BaseModel):
    """Per-model breakdown row in the stats response."""

    model_version: str
    n: int
    agreement_pct: float


class FeedbackStatsResponse(BaseModel):
    """Aggregated stats across all stored feedback."""

    total: int = Field(..., description="Total number of feedback entries.")
    agreement_rate: Optional[float] = Field(
        None,
        description="Fraction of feedback where user agreed with the model (0-1).",
    )
    per_model: list[FeedbackStatsPerModel] = Field(
        default_factory=list,
        description="Breakdown by model version.",
    )


# ===========================================================================
# Admin login (gates the Monitor dashboard in the frontend)
# ===========================================================================


class AdminLoginRequest(BaseModel):
    """Shape of the JSON body for POST /admin/login."""

    username: str = Field(..., min_length=1, description="Admin username.")
    password: str = Field(..., min_length=1, description="Admin password.")


class AdminLoginResponse(BaseModel):
    """Shape of the JSON response from POST /admin/login."""

    ok: bool = True
    token: str = Field(..., description="Signed token to send as a Bearer header.")
    expires_at: int = Field(..., description="Unix timestamp when the token expires.")
