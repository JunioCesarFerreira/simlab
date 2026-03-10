from fastapi import Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED
from pylib.service_settings import ApiAuthSettings

API_KEY = ApiAuthSettings.from_env().api_key
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )
