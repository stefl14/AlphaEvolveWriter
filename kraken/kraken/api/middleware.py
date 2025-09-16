"""Middleware for the Kraken API."""

import time
import logging
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests and their processing time."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log information."""
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Time: {process_time:.3f}s"
        )

        return response


class HealthCheckMiddleware:
    """Middleware for health check monitoring."""

    def __init__(self, app):
        """Initialize the middleware."""
        self.app = app
        self.healthy = True
        self.ready = False

    async def __call__(self, scope, receive, send):
        """Process the request."""
        await self.app(scope, receive, send)

    def set_ready(self, ready: bool):
        """Set the ready status."""
        self.ready = ready
        logger.info(f"Ready status set to: {ready}")

    def set_healthy(self, healthy: bool):
        """Set the health status."""
        self.healthy = healthy
        logger.info(f"Health status set to: {healthy}")