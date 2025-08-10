
import os
from app import create_app
import logging

app = create_app()
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')

    port = int(os.environ.get('FLASK_RUN_PORT'))

    debug_mode = app.config.get('DEBUG', False)

    logger.info(f"Starting Flask server on {host}:{port} (Debug: {debug_mode})")

    app.run(host=host, port=port, debug=debug_mode)