import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Logging configuration constants
LOG_DIR = 'logs'
LOG_FILENAME = 'ai_service.log'
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

def setup_logging():
    """
    Sets up logging to both console and file (in production),
    prevents duplicate handlers in Flask auto-reloads,
    and suppresses noise from third-party libraries.
    """
    try:
        is_development = os.environ.get('FLASK_ENV', 'production').lower() == 'development'
        log_level_name = 'DEBUG' if is_development else 'INFO'
        log_level = getattr(logging, log_level_name, logging.INFO)

        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        root_logger = logging.getLogger()

        if root_logger.hasHandlers():
            root_logger.handlers.clear()  # Prevents duplicated logs on Flask reload

        root_logger.setLevel(log_level)

        # Console logging is always enabled
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

        # Enable file logging in production environments
        if not is_development:
            try:
                log_path = os.path.join(os.getcwd(), LOG_DIR)
                os.makedirs(log_path, exist_ok=True)
                log_file_path = os.path.join(log_path, LOG_FILENAME)

                file_handler = RotatingFileHandler(
                    log_file_path,
                    maxBytes=MAX_BYTES,
                    backupCount=BACKUP_COUNT,
                    encoding='utf-8'
                )
                file_handler.setFormatter(log_formatter)
                root_logger.addHandler(file_handler)
                root_logger.info(f"File logging enabled at: {log_file_path}")
            except PermissionError:
                root_logger.error(f"Permission denied for log directory: {log_path}")
            except Exception as e:
                root_logger.error(f"Failed to configure file logging: {e}", exc_info=True)
        else:
            root_logger.info("Development mode: file logging disabled.")

        # Suppress noisy logs from common libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("peft").setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("faiss").setLevel(logging.INFO)

        root_logger.info(f"Logging configured. Level: {log_level_name}")

    except Exception as e:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.critical("Failed to configure advanced logging. Using fallback config.", exc_info=True)