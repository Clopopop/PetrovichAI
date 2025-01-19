"""
Sets up logging for the entire application.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="bot_log.log",
    filemode="a"
)

logger = logging.getLogger("PetrovichAI")
