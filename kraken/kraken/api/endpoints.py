"""API endpoints for text completion."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from kraken.api.models import (
    CompletionRequest,
    CompletionResponse,
    BatchCompletionRequest,
    BatchCompletionResponse,
    ErrorResponse
)
from kraken.ml import ModelConfig, ModelInference

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1", tags=["completion"])

# Initialize model (would normally be done at startup)
model_config = ModelConfig()
inference_engine = ModelInference(model_config)


@router.post("/complete", response_model=CompletionResponse)
async def complete_text(request: CompletionRequest):
    """Generate text completion for a single prompt."""
    try:
        # Generate completion
        result = inference_engine.generate_completion(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )

        return CompletionResponse(**result)

    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/complete/batch", response_model=BatchCompletionResponse)
async def complete_text_batch(request: BatchCompletionRequest):
    """Generate text completions for multiple prompts."""
    try:
        completions = []

        for prompt in request.prompts:
            result = inference_engine.generate_completion(
                prompt=prompt,
                max_length=request.max_length,
                temperature=request.temperature
            )
            completions.append(CompletionResponse(**result))

        return BatchCompletionResponse(
            completions=completions,
            total_prompts=len(request.prompts)
        )

    except Exception as e:
        logger.error(f"Error generating batch completions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    return {
        "model_name": model_config.model_name,
        "max_length": model_config.max_length,
        "device": inference_engine.device if inference_engine else "not_loaded"
    }