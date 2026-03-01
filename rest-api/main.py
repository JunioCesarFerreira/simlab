from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html

import os, sys
project_path = os.path.abspath(os.path.join(os.getcwd(), "..")) 
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    
from api.router import api_router

app = FastAPI(
    title="Simulation Management API",
    version="1.2.0"
)
app.include_router(api_router, prefix="/api/v1")

@app.get("/docs", include_in_schema=False)
async def custom_docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="SimLab API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)