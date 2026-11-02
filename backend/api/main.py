from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import get_settings
from backend.core.logging import setup_logging
import logging

logger = logging.getLogger("aml")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info("Starting AML Simulation Platform backend")
    yield
    # Shutdown
    logger.info("Shutting down AML Simulation Platform backend")

settings = get_settings()

app = FastAPI(
    title="AML Simulation Platform",
    description="Explainable, Multi-Agent AML Simulation and Risk Investigation API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "aml-playground"}

# Router stubs for future prompts
# from backend.api.routes import simulation, transactions, alerts, investigations, models, metrics, stream
# app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["Simulation"])