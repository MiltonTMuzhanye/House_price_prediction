from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from src.house_price_prediction.utils.logger import logger
import uvicorn

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using trained models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting House Price Prediction API...")
    logger.info("API documentation available at /docs")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down House Price Prediction API...")

@app.get("/")
async def root():
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )