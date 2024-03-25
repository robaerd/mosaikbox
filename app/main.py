from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import mix_routes, song_routes, survey_routes
from fastapi.responses import JSONResponse

from app.exceptions.custom_exceptions import NotFoundException, TaskCancelledException

import logging

# logging configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# disable numba logs
logging.getLogger('numba').setLevel(logging.WARNING)

# FastAPI configuration
app = FastAPI(title="Mosaikbox API", version="1.0.0")

# Set up CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.exception_handler(NotFoundException)
async def item_not_found_exception_handler(request: Request, exc: NotFoundException):
    return JSONResponse(
        status_code=404,
        content={"message": f"{exc.object_type} with ID {exc.object_id} not found"},
    )


@app.exception_handler(TaskCancelledException)
async def task_cancelled_exception_handler(request: Request, exc: TaskCancelledException):
    return JSONResponse(
        status_code=202,
        content={"message": f"{exc.message}"},
    )


# Include routers (API endpoints)
app.include_router(mix_routes.router)
app.include_router(song_routes.router)
app.include_router(survey_routes.router)


@app.get("/")
def read_root():
    return {"message": "Welcome to Mosaikbox!"}
