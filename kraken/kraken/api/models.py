"""Pydantic models for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """Request model for text completion."""

    prompt: str = Field(
        ...,
        description="The text prompt to complete",
        min_length=1,
        max_length=1000
    )
    max_length: Optional[int] = Field(
        100,
        description="Maximum length of the completion",
        ge=10,
        le=500
    )
    temperature: Optional[float] = Field(
        0.7,
        description="Sampling temperature for text generation",
        ge=0.1,
        le=2.0
    )
    top_p: Optional[float] = Field(
        0.9,
        description="Top-p sampling parameter",
        ge=0.1,
        le=1.0
    )


class CompletionResponse(BaseModel):
    """Response model for text completion."""

    prompt: str = Field(..., description="Original prompt")
    completion: str = Field(..., description="Generated text completion")
    total_tokens: int = Field(..., description="Total tokens processed")
    model_name: str = Field(..., description="Model used for generation")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")