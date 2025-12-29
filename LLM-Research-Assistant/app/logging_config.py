"""
logging_config.py
Application-wide logging setup.
"""

import logging
from app.config import LOG_FILE

def setup_logging():
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler (writes to logs/app.log)
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Console handler (prints to PowerShell / terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(module)s — %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers if not already added
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        # Avoid duplicate handlers during streamlit reloads
        logger.handlers = [file_handler, console_handler]

    logging.info("Logging initialized successfully.")
