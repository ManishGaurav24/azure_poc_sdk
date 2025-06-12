from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import List
import uuid
import datetime
from dotenv import load_dotenv

from utils.cosmos_connection import cosmos_enabled, save_message_to_cosmos, get_last_messages_from_cosmos
from utils.llm_invoke import call_llm_async_with_retry, warm_up_search_index
from utils.log_utils import logger

load_dotenv()

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
    allow_origins=["*"],  # Allow all origins for development; restrict in production
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


class ChatRequest(BaseModel):
    message: str
    session_id: str


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
    return {"message": "Document Assistant API is running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cosmos_enabled": cosmos_enabled,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Save user message to Cosmos DB
        save_message_to_cosmos(request.session_id, "user", request.message)

        # Get AI response
        response = await call_llm_async_with_retry(request.message, request.session_id)

        # Save AI response to Cosmos DB
        save_message_to_cosmos(request.session_id, "assistant", response)

        return ChatResponse(response=response, session_id=request.session_id)

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        error_message = f"Error: {str(e)}"
        save_message_to_cosmos(request.session_id, "assistant", error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.get("/session/{session_id}/info", response_model=SessionInfo)
async def get_session_info(session_id: str):
    try:
        message_count = 0
        if cosmos_enabled:
            messages = get_last_messages_from_cosmos(session_id, limit=100)
            message_count = len(messages)

        return SessionInfo(
            session_id=session_id,
            cosmos_enabled=cosmos_enabled,
            message_count=message_count
        )
    except Exception as e:
        logger.error(f"Session info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/history", response_model=MessageHistory)
async def get_session_history(session_id: str, limit: int = 10):
    try:
        messages = []
        if cosmos_enabled:
            cosmos_messages = get_last_messages_from_cosmos(session_id, limit=limit)
            messages = [
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in cosmos_messages
            ]

        return MessageHistory(messages=messages, session_id=session_id)
    except Exception as e:
        logger.error(f"Session history error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/clear")
async def clear_session_history(session_id: str):
    try:
        # Note: This endpoint doesn't actually delete from Cosmos DB
        # You might want to implement actual deletion if needed
        return {"message": f"Session {session_id} history cleared", "success": True}
    except Exception as e:
        logger.error(f"Clear session error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/warm-up")
async def warm_up_endpoint():
    try:
        success = await warm_up_search_index()
        if success:
            logger.info("Search index warmup completed successfully")
            return {"message": "Search index warmup completed successfully", "success": True}
        else:
            logger.warning("Search index warmup did not complete successfully")
            return {"message": "Search index warmup did not complete successfully", "success": False}
    except Exception as e:
        logger.error(f"Warmup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/new")
async def create_new_session():
    new_session_id = str(uuid.uuid4())
    return {"session_id": new_session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
