from flask import Blueprint, request, jsonify, current_app
import logging
from ..services import shap_service

logger = logging.getLogger(__name__)
explain_bp = Blueprint('explain_bp', __name__)

@explain_bp.route('/', methods=['POST'])
def handle_explanation():
    """Handles POST requests to generate SHAP explanations for a given input text."""
    if not request.is_json:
        logger.warning("Explain API: Invalid request format: Not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text_to_explain = data.get('text')
    session_id = data.get('session_id')

    if not text_to_explain or not isinstance(text_to_explain, str):
        logger.warning("Explain API: Missing or invalid 'text' field")
        return jsonify({"error": "Missing or invalid 'text' field for explanation"}), 400

    logger.info(f"Explanation request received. Session: {session_id}. Text length: {len(text_to_explain)}")

    try:
        # Generate SHAP explanation
        explanation_result = shap_service.get_explanation(text_to_explain)

        if explanation_result.get("status") == "success":
            logger.info(f"Explain API: SHAP explanation generated successfully for session {session_id}.")
            return jsonify({"explanation": explanation_result.get("data")}), 200
        else:
            error_msg = explanation_result.get("message", "Unknown explanation error")
            logger.warning(f"Explain API: SHAP explanation failed for session {session_id}: {error_msg}")
            return jsonify({"error": "Explanation generation failed", "details": error_msg}), 500

    except Exception as e:
        logger.error(f"Unexpected error during explanation request for session {session_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during explanation process"}), 500
