import os
import logging
import sys
from flask import Flask, jsonify
from dotenv import load_dotenv

# Load environment variables from the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=dotenv_path)

from .config import Config
from .utils.logging_config import setup_logging
setup_logging()

from .services import llm_service, vector_store_service, shap_service

logger = logging.getLogger(__name__)

def create_app():
    """Flask application factory"""
    app = Flask(__name__)
    app.config.from_object(Config)
    logger.info(f"Flask AI application starting. Env: {app.config['FLASK_ENV']}.")

    # Validate environment variables and file paths
    try:
        Config.validate()
        logger.info("Configuration validated successfully.")
    except ValueError as e:
        logger.critical(f"CONFIG VALIDATION ERROR: {e}. Aborting application startup.")
        sys.exit(1)

    # Initialize vector store for RAG features
    try:
        vector_init_success = vector_store_service.initialize_vector_store(
            index_path=app.config['FAISS_INDEX_PATH'],
            data_path=app.config['DISEASE_DATA_PATH'],
            map_path=app.config['ID_MAP_PATH'],
            model_name=app.config['EMBEDDING_MODEL_NAME']
        )
        if not vector_init_success:
            logger.warning("Vector store service failed to initialize. Retrieval may be unavailable.")
        else:
            logger.info("Vector store service initialized.")
    except Exception as e:
        logger.error(f"CRITICAL ERROR during vector store initialization: {e}", exc_info=True)
        # Uncomment below if vector store is essential
        # sys.exit(1)

    # Load language model with optional adapters
    try:
        adapter_id_or_path = app.config.get('STUDENT_MODEL_ADAPTERS_PATH')
        llm_service.load_model(
            model_name=app.config['LLM_MODEL_NAME'],
            device_pref=app.config['LLM_DEVICE'],
            adapter_id_or_path=adapter_id_or_path
        )
        if not llm_service.is_model_loaded():
            logger.error("LLM failed to load. Core functionality may be broken.")
        else:
            logger.info("LLM service initialized.")
    except Exception as e:
        logger.error(f"CRITICAL ERROR during LLM loading: {e}", exc_info=True)
        sys.exit(1)

    # Log availability of SHAP explanations
    if shap_service.shap_available:
        logger.info("SHAP is available and will be lazily initialized.")
    else:
        logger.warning("SHAP not available. Explanation endpoints will be disabled.")

    # Register API routes
    logger.info("Registering API blueprints...")
    from .api.reason_api import reason_bp
    from .api.explain_api import explain_bp

    app.register_blueprint(reason_bp, url_prefix='/api/reason')
    app.register_blueprint(explain_bp, url_prefix='/api/explain')
    logger.info("Blueprints registered.")

    # Health check endpoint
    @app.route('/health')
    def health():
        vector_ok = vector_store_service.get_status()
        llm_ok = llm_service.is_model_loaded()
        is_healthy = vector_ok and llm_ok
        status_code = 200 if is_healthy else 503

        return jsonify({
            "status": "ok" if is_healthy else "error",
            "services": {
                "vector_store": "initialized" if vector_ok else "not_initialized",
                "language_model": "loaded" if llm_ok else "not_loaded",
                "explainability (shap)": "available" if shap_service.shap_available else "unavailable"
            }
        }), status_code

    logger.info("Flask AI application created and configured.")
    return app
