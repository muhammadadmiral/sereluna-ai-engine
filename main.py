import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(message)s",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import (
    account_router,
    article_router,
    auth_router,
    calendar_router,
    chat_router,
    context_router,
    device_router,
    diary_router,
    gamification_router,
    media_router,
    mood_router,
    model_router,
    notification_router,
    profile_router,
    screening_router,
    sleep_router,
    statistics_router,
    stats_router,
)

app = FastAPI(title="Sereluna AI Engine", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    # Warm up the ML models for the Professor Demo
    from services.nlp.ml_service import get_trained_model
    import logging
    logger = logging.getLogger("sereluna.startup")
    logger.info("Initializing Machine Learning Models (Full Data Mining Pipeline)...")
    try:
        # This will trigger the 5-fold cross-validation once
        get_trained_model()
        logger.info("ML Models initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ML models: {e}")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat_router.router)
app.include_router(gamification_router.router)
app.include_router(article_router.router, prefix="/api/v1")
app.include_router(auth_router.router)
app.include_router(account_router.router, prefix="/api/v1")
app.include_router(screening_router.router)
app.include_router(context_router.router)
app.include_router(profile_router.router, prefix="/api/v1")
app.include_router(diary_router.router, prefix="/api/v1")
app.include_router(notification_router.router, prefix="/api/v1")
app.include_router(media_router.router, prefix="/api/v1")
app.include_router(device_router.router, prefix="/api/v1")
app.include_router(sleep_router.router, prefix="/api/v1")
app.include_router(mood_router.router, prefix="/api/v1")
app.include_router(model_router.router)
app.include_router(calendar_router.router, prefix="/api/v1")
app.include_router(stats_router.router, prefix="/api/v1")
app.include_router(statistics_router.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Sereluna AI Engine is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
