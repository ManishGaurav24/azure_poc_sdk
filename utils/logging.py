import logging
import json


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_print(message: str, data=None):
    """Debug print function to log information"""
    logger.info(f"[DEBUG] {message}")
    if data:
        logger.info(f"[DEBUG] Data: {json.dumps(data, indent=2, default=str)}")