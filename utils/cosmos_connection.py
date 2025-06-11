import datetime
import os
import uuid
from azure.cosmos import CosmosClient

from utils.logging import logger, debug_print

COSMOS_CONNECTION_STRING = os.getenv("COSMOS_CONNECTION_STRING")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME", "chatdb")
COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME", "messages")

# Initialize Cosmos DB client
try:
    cosmos_client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
    database = cosmos_client.get_database_client(COSMOS_DB_NAME)
    container = database.get_container_client(COSMOS_CONTAINER_NAME)
    cosmos_enabled = True
    logger.info("Cosmos DB connection established successfully")
except Exception as e:
    logger.error(f"Cosmos DB connection failed: {str(e)}")
    cosmos_enabled = False


def save_message_to_cosmos(session_id: str, role: str, content: str):
    """Save a message to Cosmos DB"""
    if not cosmos_enabled:
        debug_print("Cosmos DB not enabled, skipping message save")
        return

    try:
        item = {
            "id": str(uuid.uuid4()),
            "sessionId": session_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "role": role,
            "content": content
        }
        container.create_item(body=item)
        debug_print(f"Message saved to Cosmos DB", {"role": role, "content_length": len(content)})
    except Exception as e:
        debug_print(f"Failed to save message to Cosmos DB: {str(e)}")


def get_last_messages_from_cosmos(session_id: str, limit: int = 5):
    """Fetch last N messages for context from Cosmos DB"""
    if not cosmos_enabled:
        debug_print("Cosmos DB not enabled, returning empty context")
        return []

    try:
        query = f"""
        SELECT TOP {limit} * FROM c
        WHERE c.sessionId = @sessionId
        ORDER BY c.timestamp DESC
        """
        parameters = [{"name": "@sessionId", "value": session_id}]
        items = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        # Sort oldest to newest for conversation flow
        sorted_items = sorted(items, key=lambda x: x["timestamp"])
        debug_print(f"Retrieved {len(sorted_items)} messages from Cosmos DB for session {session_id}")
        return sorted_items
    except Exception as e:
        debug_print(f"Failed to retrieve messages from Cosmos DB: {str(e)}")
        return []
