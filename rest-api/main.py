from fastapi import FastAPI

import os, sys
project_path = os.path.abspath(os.path.join(os.getcwd(), "..")) 
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    
from api.router import api_router

app = FastAPI(title="Simulation Management API")
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)