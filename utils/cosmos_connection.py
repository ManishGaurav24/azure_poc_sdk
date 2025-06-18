import datetime
import os
from functools import partial

from dotenv import load_dotenv
import uuid
from azure.cosmos import CosmosClient

from utils.log_utils import logger, debug_print
load_dotenv()

COSMOS_CONNECTION_STRING = os.getenv("COSMOS_CONNECTION_STRING")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME")

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


def save_message_to_cosmos(session_id: str, user_id:str, user_roles:list[str], role: str, content: str):
    """Save a message to Cosmos DB"""
    if not cosmos_enabled:
        debug_print("Cosmos DB not enabled, skipping message save")
        return
    try:
        item = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "user_id": user_id,
            "userRoles": user_roles,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "role": role,
            "content": content
        }
        container.create_item(body=item)
        debug_print(f"Message saved to Cosmos DB", {"role": role, "content_length": len(content)})
    except Exception as e:
        debug_print(f"Failed to save message to Cosmos DB: {str(e)}")


def get_latest_session_ids(user_id: str, limit: int = 5):
    if not cosmos_enabled:
        debug_print("Cosmos DB not enabled, returning None for session ID")
        return None
    try:
        query = """
        SELECT c.session_id, c.timestamp FROM c
        WHERE c.user_id = @user_id
        ORDER BY c.timestamp DESC
        """
        parameters = [{"name": "@user_id", "value": user_id}]

        items = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        # Deduplicate and limit
        seen = set()
        unique_sessions = []
        for item in items:
            sid = item["session_id"]
            if sid not in seen:
                seen.add(sid)
                unique_sessions.append(sid)
            if len(unique_sessions) >= limit:
                break

        return unique_sessions

    except Exception as e:
        debug_print(f"Failed to retrieve latest session IDs: {str(e)}")
        return None


def get_last_messages_from_cosmos(user_id: str, limit: int = 5):
    """Fetch last N messages per session from Cosmos DB"""
    if not cosmos_enabled:
        debug_print("Cosmos DB not enabled, returning empty context")
        return []

    session_ids = get_latest_session_ids(user_id=user_id, limit=limit)
    if not session_ids:
        debug_print("No session IDs found, returning empty context")
        return []

    messages = []
    for session_id in session_ids:
        try:
            query = f"""
            SELECT c.id, c.role, c.content, c.timestamp FROM c
            WHERE c.user_id = @user_id AND c.session_id = @session_id
            ORDER BY c.timestamp DESC
            """
            parameters = [
                {"name": "@user_id", "value": user_id},
                {"name": "@session_id", "value": session_id}
            ]
            items = list(container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            sorted_items = sorted(items, key=lambda x: x["timestamp"])
            debug_print(f"Retrieved {len(sorted_items)} messages from Cosmos DB for session {session_id}")

            # Append a dict for each session
            messages.append({
                "session_id": session_id,
                "messages": sorted_items
            })

        except Exception as e:
            debug_print(f"Failed to retrieve messages from Cosmos DB for session {session_id}: {str(e)}")
            continue

    return messages
