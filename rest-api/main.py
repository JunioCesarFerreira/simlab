from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager

import os, sys
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from api.router import api_router
from api.responses import SafeJSONResponse
from api.dependencies import close_factory


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        close_factory()

app = FastAPI(
    title="Simulation Management API",
    version="2.0.0",
    openapi_version="3.0.3",
    # Tolerate non-finite floats (inf/nan) in stored objectives so reading an
    # experiment never fails with HTTP 500 during JSON serialization.
    default_response_class=SafeJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=4)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health", tags=["health"], include_in_schema=False)
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
