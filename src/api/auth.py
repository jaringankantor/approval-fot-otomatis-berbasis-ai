import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.api.settings import API_KEY, API_KEY_HEADER_NAME


api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False
)


def require_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "API_KEY belum dikonfigurasi di file .env."
            }
        )

    if not api_key or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "error": "API key tidak valid atau tidak dikirim."
            }
        )

    return True
