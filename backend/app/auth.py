"""Simple API key authentication and in-memory rate limiter."""
import time
import threading
from typing import Optional

from fastapi import Header, HTTPException, Request, status

from shared.config import settings
import logging

logger = logging.getLogger(__name__)

# Simple in-memory rate limiter state
_rate_state = {}
_rate_lock = threading.Lock()


def _parse_authorization(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    # support 'Bearer <token>' or raw token
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == 'bearer':
        return parts[1]
    return authorization


def _check_rate_limit(api_key: str):
    if not settings.RATE_LIMIT_ENABLED:
        return

    limit = settings.RATE_LIMIT_PER_MIN
    window = 60
    now = int(time.time())
    window_start = now - (now % window)

    with _rate_lock:
        state = _rate_state.get(api_key)
        if state is None or state['window_start'] != window_start:
            _rate_state[api_key] = {'window_start': window_start, 'count': 1}
            return

        if state['count'] >= limit:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail=f"Rate limit exceeded ({limit}/min)")
        state['count'] += 1


async def get_api_key(authorization: Optional[str] = Header(None), request: Request = None) -> str:
    """FastAPI dependency to validate API key and apply rate limiting."""
    api_key = _parse_authorization(authorization)
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    # Validate key
    if api_key != settings.API_KEY:
        logger.warning(f"Invalid API key from {request.client.host if request and request.client else 'unknown'}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")

    # Rate limit
    try:
        _check_rate_limit(api_key)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Rate limiter error: {e}")

    return api_key
