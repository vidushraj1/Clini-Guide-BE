import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configures application-wide logging based on FLASK_ENV."""
    try:
        # Use DEBUG level in development, INFO otherwise
        log_level_name = os.environ.get('FLASK_ENV', 'production').lower() == 'development' and 'DEBUG' or 'INFO'
        log_level = getattr(logging, log_level_name, logging.INFO)

        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        root_logger = logging.getLogger()

        # Avoid duplicate logs on hot reload (e.g., in Flask dev server)
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        root_logger.setLevel(log_level)

        # Always log to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

        # In non-dev environments, also write logs to rotating file
        if os.environ.get('FLASK_ENV') != 'development':
            try:
                log_dir = 'logs'
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, 'ai_service.log')
                file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
                file_handler.setFormatter(log_formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                root_logger.error(f"Failed to set up file logging to '{log_file}': {e}", exc_info=True)

        # Reduce noise from overly verbose libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.INFO)
        logging.getLogger("peft").setLevel(logging.INFO)
        logging.getLogger("trl").setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)

        root_logger.info(f"Logging configured. Level: {log_level_name}")

    except Exception as e:
        # Fallback if logging setup fails early in runtime
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.error(f"Error setting up custom logging: {e}. Using basic config.", exc_info=True)
