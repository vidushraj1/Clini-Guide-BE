import faiss
import numpy as np
import json
import logging
import os
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Global variables for the loaded store
faiss_index = None
disease_data_store = None
faiss_id_to_disease_map = None
embedding_model = None
embedding_dim = None
is_initialized = False

def initialize_vector_store(index_path: str, data_path: str, map_path: str, model_name: str):
    """Loads the FAISS index, data store, ID map, and embedding model."""
    global faiss_index, disease_data_store, faiss_id_to_disease_map, embedding_model, embedding_dim, is_initialized

    if is_initialized:
        logger.warning("Vector store already initialized. Skipping.")
        return True

    logger.info("Initializing Vector Store Service...")
    load_success = True

    # Load Embedding Model
    try:
        logger.info(f"Loading embedding model: {model_name}...")
        embedding_model = SentenceTransformer(model_name)
        test_emb = embedding_model.encode("test")
        embedding_dim = len(test_emb)
        logger.info(f"Embedding model loaded. Dimension: {embedding_dim}")
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
        embedding_model = None
        load_success = False

    # Load FAISS Index
    if load_success:
        try:
            if not os.path.exists(index_path):
                 raise FileNotFoundError(f"FAISS index file not found at {index_path}")
            logger.info(f"Loading FAISS index from: {index_path}")
            faiss_index = faiss.read_index(index_path)
            if faiss_index.d != embedding_dim:
                logger.error(f"FAISS index dimension ({faiss_index.d}) does not match model dimension ({embedding_dim})!")
                faiss_index = None
                load_success = False
            else:
                logger.info(f"FAISS index loaded successfully. Contains {faiss_index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}", exc_info=True)
            faiss_index = None
            load_success = False

    # Load Disease Data Store
    if load_success:
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Disease data file not found at {data_path}")
            logger.info(f"Loading disease data store from: {data_path}")
            with open(data_path, 'r') as f:
                disease_data_store = json.load(f)
            logger.info(f"Disease data store loaded. Contains {len(disease_data_store)} entries.")
        except Exception as e:
            logger.error(f"Failed to load disease data store: {e}", exc_info=True)
            disease_data_store = None
            load_success = False

    # Load FAISS ID to Disease Map
    if load_success:
        try:
            if not os.path.exists(map_path):
                 raise FileNotFoundError(f"ID map file not found at {map_path}")
            logger.info(f"Loading FAISS ID to disease map from: {map_path}")
            with open(map_path, 'r') as f:
                loaded_map = json.load(f)
                faiss_id_to_disease_map = {int(k): v for k, v in loaded_map.items()}
            logger.info(f"FAISS ID map loaded. Contains {len(faiss_id_to_disease_map)} mappings.")
            if faiss_index and faiss_index.ntotal != len(faiss_id_to_disease_map):
                 logger.warning(f"Mismatch: FAISS index size ({faiss_index.ntotal}) vs ID map size ({len(faiss_id_to_disease_map)})")

        except Exception as e:
            logger.error(f"Failed to load FAISS ID map: {e}", exc_info=True)
            faiss_id_to_disease_map = None
            load_success = False

    if load_success:
        is_initialized = True
        logger.info("Vector Store Service initialized successfully.")
        return True
    else:
        faiss_index = None
        disease_data_store = None
        faiss_id_to_disease_map = None
        embedding_model = None
        embedding_dim = None
        is_initialized = False
        logger.error("Vector Store Service initialization failed.")
        return False


def get_status():
    """Returns the initialization status."""
    return is_initialized

def search_diseases(query_text: str, k: int = 3) -> list[tuple[str, float]]:
    """
    Embeds the query text and searches the FAISS index for the top k similar diseases.
    Returns a list of tuples: (disease_name, similarity_score)
    """
    if not is_initialized or not faiss_index or not embedding_model or not faiss_id_to_disease_map:
        logger.error("Vector store not initialized. Cannot perform search.")
        return []

    try:
        logger.debug(f"Embedding query for search: '{query_text[:100]}...'")
        query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)

        query_embedding_normalized = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding_normalized)

        logger.debug(f"Searching FAISS index for top {k} results...")
        distances, indices = faiss_index.search(query_embedding_normalized, k)

        results = []
        if len(indices) > 0:
            for i in range(len(indices[0])):
                faiss_id = indices[0][i]
                if faiss_id == -1:
                     continue
                distance = distances[0][i]
                similarity_score = 1.0 - (distance**2 / 2.0)

                if faiss_id in faiss_id_to_disease_map:
                    disease_name = faiss_id_to_disease_map[faiss_id]
                    results.append((disease_name, float(similarity_score)))
                else:
                    logger.warning(f"FAISS index {faiss_id} not found in ID map.")

        logger.info(f"FAISS search returned {len(results)} diseases for query.")
        return results

    except Exception as e:
        logger.error(f"Error during FAISS search: {e}", exc_info=True)
        return []

def get_disease_data(disease_name: str) -> dict | None:
    """Retrieves the full data dictionary for a given disease name."""
    if not is_initialized or not disease_data_store:
        logger.error("Vector store not initialized or data store not loaded.")
        return None

    data = disease_data_store.get(disease_name)
    if not data:
        logger.warning(f"Data for disease '{disease_name}' not found in the store.")
    return data