import os
import torch
from dotenv import load_dotenv

# Load .env variables from project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=dotenv_path)

class Config:
    """Configuration settings loaded from environment variables"""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-!@#')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    DEBUG = FLASK_ENV == 'development'

    # LLM model configuration
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME')
    LLM_DEVICE = os.environ.get('LLM_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    STUDENT_MODEL_ADAPTERS_PATH = os.environ.get('STUDENT_MODEL_ADAPTERS_PATH')

    # Embedding model & RAG-related paths
    EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
    VECTOR_DATA_DIR = os.path.join(BASE_DIR, os.environ.get('VECTOR_DATA_OUTPUT_DIR', 'vector_data_store'))
    FAISS_INDEX_PATH = os.path.join(VECTOR_DATA_DIR, os.environ.get('FAISS_INDEX_FILENAME', 'disease_index.faiss'))
    DISEASE_DATA_PATH = os.path.join(VECTOR_DATA_DIR, os.environ.get('DISEASE_DATA_FILENAME', 'disease_data.json'))
    ID_MAP_PATH = os.path.join(VECTOR_DATA_DIR, os.environ.get('ID_MAP_FILENAME', 'faiss_id_to_disease.json'))
    RAG_TOP_K = int(os.environ.get('RAG_TOP_K', 3))

    # Optional backend API for collecting user feedback
    NODE_BACKEND_URL = os.environ.get('NODE_BACKEND_URL', 'http://localhost:3000')
    FEEDBACK_API_ENDPOINT = os.environ.get('FEEDBACK_API_ENDPOINT', f'{NODE_BACKEND_URL}/api/feedback/all')
    NODE_API_KEY = os.environ.get('NODE_API_KEY')

    @classmethod
    def validate(cls):
        """Checks that essential environment variables and file paths exist"""
        required = [
            'LLM_MODEL_NAME', 'LLM_DEVICE', 'EMBEDDING_MODEL_NAME',
            'FAISS_INDEX_PATH', 'DISEASE_DATA_PATH', 'ID_MAP_PATH'
        ]

        # Only require backend API key if endpoint is not localhost
        if cls.FEEDBACK_API_ENDPOINT and 'localhost:3000' not in cls.FEEDBACK_API_ENDPOINT:
            required.append('NODE_API_KEY')

        missing = []
        for var in required:
            value = getattr(cls, var, None)

            if not value:
                missing.append(var)

            elif var.endswith('_PATH') and not os.path.exists(value):
                # STUDENT_MODEL_ADAPTERS_PATH is optional and may not exist
                if var == 'STUDENT_MODEL_ADAPTERS_PATH' and not value:
                    continue
                elif var != 'STUDENT_MODEL_ADAPTERS_PATH':
                    missing.append(f"{var} (Path not found: {value})")

        if missing:
            raise ValueError(
                f"Missing/Invalid critical config vars (check .env & paths): {', '.join(missing)}"
            )
