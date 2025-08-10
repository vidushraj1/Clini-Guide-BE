import torch
import logging
import numpy as np

from . import llm_service

# Conditional SHAP import
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False

from peft import PeftModel

logger = logging.getLogger(__name__)
explainer = None
is_initialized = False

def initialize_explainer():
    """Initializes the SHAP explainer if not already initialized."""
    global explainer, is_initialized

    if not shap_available:
        logger.warning("SHAP is not installed. Explanation functionality disabled.")
        return False

    if is_initialized:
        return True

    if not llm_service.is_model_loaded():
        logger.error("Cannot initialize SHAP explainer: LLM is not loaded.")
        return False

    current_model = llm_service.model
    current_tokenizer = llm_service.tokenizer

    try:
        model_for_shap = current_model
        if hasattr(current_model, 'base_model') and hasattr(current_model.base_model, 'model'):
            model_for_shap = current_model.base_model.model
        elif hasattr(current_model, 'model') and not isinstance(current_model, PeftModel):
            model_for_shap = current_model.model

        logger.info("Initializing SHAP Partition Explainer...")
        explainer = shap.Explainer(model_for_shap, current_tokenizer, algorithm="partition")
        is_initialized = True
        logger.info("SHAP explainer initialized.")
        return True
    except Exception as e:
        logger.error(f"SHAP initialization failed: {e}", exc_info=True)
        explainer = None
        is_initialized = False
        return False

def get_explanation(text_input: str, max_evals: int = 50) -> dict:
    """
    Generates a SHAP explanation for the given text input.
    Returns a dictionary with token-level importance scores.
    """
    global explainer, is_initialized

    if not is_initialized and not initialize_explainer():
        return {"status": "failed", "message": "SHAP explainer not available."}

    if not text_input or not isinstance(text_input, str):
        return {"status": "failed", "message": "Invalid input text."}

    logger.info(f"Generating SHAP explanation (max_evals={max_evals}). Input length: {len(text_input)}")

    if explainer is None:
        return {"status": "failed", "message": "SHAP explainer is not initialized."}

    try:
        shap_values = explainer([text_input], max_evals=max_evals, batch_size=1)

        if (hasattr(shap_values, 'data') and isinstance(shap_values.data, (list, np.ndarray)) and
            hasattr(shap_values, 'values') and isinstance(shap_values.values, (list, np.ndarray)) and
            shap_values.data[0] is not None and shap_values.values[0] is not None):

            tokens = list(shap_values.data[0])
            scores_raw = shap_values.values[0]

            # Handle multidimensional score arrays
            if isinstance(scores_raw, (torch.Tensor, np.ndarray)) and scores_raw.ndim > 1:
                scores = np.sum(scores_raw, axis=-1).tolist()
            elif isinstance(scores_raw, (list, np.ndarray)):
                scores = list(scores_raw)
            else:
                logger.warning("Unexpected SHAP score structure.")
                scores = []

            if len(tokens) == len(scores):
                explanation = [{"token": str(t), "score": float(s)} for t, s in zip(tokens, scores)]
                logger.info(f"Generated explanation with {len(explanation)} tokens.")
                return {"status": "success", "data": {"tokens": explanation}}
            else:
                logger.warning("Token and score counts do not match.")
                return {"status": "failed", "message": "Mismatch between tokens and scores."}
        else:
            logger.warning("SHAP output structure invalid or empty.")
            return {"status": "failed", "message": "Unexpected SHAP output format."}
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}", exc_info=True)
        return {"status": "failed", "message": f"Explanation error: {e}"}
