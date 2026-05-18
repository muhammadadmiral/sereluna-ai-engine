from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat_router, context_router, screening_router

app = FastAPI(title="Sereluna AI Engine", version="1.0.0")

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
app.include_router(screening_router.router)
app.include_router(context_router.router)

@app.get("/")
async def root():
    return {"message": "Sereluna AI Engine is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
