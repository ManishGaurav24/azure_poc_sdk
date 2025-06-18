from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import List
import uuid
import datetime
from dotenv import load_dotenv
import os
import uvicorn

from utils.cosmos_connection import cosmos_enabled, save_message_to_cosmos, get_last_messages_from_cosmos
from utils.llm_invoke import call_llm_async_with_retry, warm_up_search_index
from utils.log_utils import logger

load_dotenv()

# Read and process allowed origins
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins_list = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]

app = FastAPI(
    title="AZURE AI CHATBOT API",
    version="0.0.1",
    description="API for Azure AI Chatbot with Cosmos DB integration",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime.datetime


class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: str
    user_roles: List[str] = []


class ChatResponse(BaseModel):
    response: str
    session_id: str


class SessionInfo(BaseModel):
    session_id: str
    cosmos_enabled: bool
    message_count: int


class MessageHistory(BaseModel):
    messages: List[ChatMessage]
    session_id: str


# API Endpoints
@app.get("/")
async def root():
    try:
        success = await warm_up_search_index()
        if success:
            logger.info("Search index warmup completed successfully")
            return {
                "message": "Search index warmup completed successfully",
                "status": "healthy",
                "cosmos_enabled": cosmos_enabled,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "success": True
            }
        else:
            logger.warning("Search index warmup did not complete successfully")
            return {
                "message": "Search index warmup did not complete successfully",
                "status": "healthy",
                "cosmos_enabled": cosmos_enabled,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "success": False
            }
    except Exception as e:
        logger.error(f"Warmup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/new")
async def create_new_session():
    new_session_id = str(uuid.uuid4())
    return {"session_id": new_session_id}


@app.get("/session/history")
async def get_session_history(user_id: str):
    try:
        messages = []
        if cosmos_enabled:
            messages = get_last_messages_from_cosmos(user_id=user_id)
        return messages
    except Exception as e:
        logger.error(f"Session history error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Save user message to Cosmos DB
        save_message_to_cosmos(session_id=request.session_id, user_id=request.user_id, user_roles=request.user_roles,
                               role="user", content=request.message)

        # Get AI response
        response = await call_llm_async_with_retry(request.message, request.session_id)

        # Save AI response to Cosmos DB
        save_message_to_cosmos(session_id=request.session_id, user_id=request.user_id, user_roles=request.user_roles,
                               role="assistant", content=response)

        return ChatResponse(response=response, session_id=request.session_id)

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        error_message = f"Error: {str(e)}"
        save_message_to_cosmos(session_id=request.session_id, user_id=request.user_id, user_roles=request.user_roles,
                               role="error", content=error_message)
        raise HTTPException(status_code=500, detail=error_message)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
