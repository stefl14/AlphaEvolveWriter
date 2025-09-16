"""Main FastAPI application for Kraken text completion API."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kraken.api.models import CompletionRequest, CompletionResponse
from kraken.api.endpoints import router as completion_router
from kraken.api.middleware import LoggingMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Kraken API server...")
    yield
    # Shutdown
    logger.info("Shutting down Kraken API server...")


# Create FastAPI app
app = FastAPI(
    title="Kraken Text Completion API",
    description="Intelligent text auto-completion for customer service messages",
    version="0.1.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(completion_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Kraken Text Completion API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint for container orchestration."""
    # Check if model is loaded
    from kraken.api.endpoints import inference_engine

    if inference_engine and inference_engine.model is not None:
        return {"status": "ready", "model_loaded": True}
    else:
        return {"status": "not_ready", "model_loaded": False}