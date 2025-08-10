from flask import Blueprint, request, jsonify, current_app
import logging
from ..services import llm_service, vector_store_service

logger = logging.getLogger(__name__)
reason_bp = Blueprint('reason_bp', __name__)

@reason_bp.route('/', methods=['POST'])
def handle_reasoning():
    if not request.is_json:
        logger.warning("Reason API: Invalid request: Content-Type must be JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    session_id = data.get('session_id', 'N/A')
    user_input = data.get('user_input')
    profile = data.get('profile', {})
    history = data.get('history', [])
    generate_explanation_flag = data.get("generate_explanation", False)

    # Input Validation
    if not user_input or not isinstance(user_input, str):
        logger.warning("Reason API: Missing or invalid 'user_input'")
        return jsonify({"error": "Missing or invalid 'user_input'"}), 400
    if not isinstance(profile, dict):
         logger.warning("Reason API: Invalid 'profile' format, expected dictionary")
         return jsonify({"error": "Invalid 'profile' format."}), 400
    if not isinstance(history, list):
         logger.warning("Reason API: Invalid 'history' format, expected list")
         return jsonify({"error": "Invalid 'history' format."}), 400

    logger.info(f"Reason API Request -> Session: {session_id}. Input: '{user_input[:100]}...'")

    # RAG Retrieval
    retrieved_context_str = "No relevant information found in the knowledge base for the provided input."
    rag_top_k = current_app.config.get('RAG_TOP_K', 3)

    # Check if vector store is initialized before attempting search
    if not vector_store_service.get_status():
        logger.error("Reason API: Vector store service is not initialized. Cannot perform RAG retrieval.")
    else:
        try:
            logger.info(f"Reason API: Performing RAG search for top {rag_top_k} documents...")
            search_results = vector_store_service.search_diseases(user_input, k=rag_top_k)

            if search_results:
                logger.info(f"Reason API: RAG search found: {[(name, f'{score:.3f}') for name, score in search_results]}")
                context_parts = ["Retrieved Information:"]
                for disease_name, score in search_results:
                    disease_data = vector_store_service.get_disease_data(disease_name)
                    if disease_data:
                        context_parts.append(f"\n--- Context for {disease_name} (Relevance Score: {score:.3f}) ---")
                        context_parts.append(f"Symptoms: {', '.join(disease_data.get('symptoms', ['N/A']))}")
                        meds_summary = [f"{med.get('name', '?')} ({med.get('dosage', 'N/A')})"
                                        for med in disease_data.get('medications', [])[:3]]
                        if meds_summary: context_parts.append(f"Potential Meds: {'; '.join(meds_summary)}")
                        severity = disease_data.get('severity_indicators', [])
                        if severity: context_parts.append(f"Severity Warnings: {', '.join(severity)}")
                    else:
                         logger.warning(f"Reason API: Could not retrieve full data for disease '{disease_name}' found in search.")
                retrieved_context_str = "\n".join(context_parts)
            else:
                logger.info("Reason API: RAG search returned no relevant results.")

        except Exception as e:
            logger.error(f"Reason API: Error during RAG retrieval step: {e}", exc_info=True)
            retrieved_context_str = "Error retrieving information from the knowledge base."

    logger.debug(f"Reason API: Retrieved Context Snippet:\n{retrieved_context_str[:500]}...")

    # Chain-of-Thought Prompt Construction
    system_message = (
        "You are 'Medi', an AI medical assistant. Your primary goal is patient safety and providing clear information based *only* on the context provided.\n"
        "1. Analyze the User Input, Conversation History, and the Retrieved Information below.\n"
        "2. Follow the Chain-of-Thought Process: Start with 'REASONING:' outlining your step-by-step thinking based *strictly* on the provided context. Consider symptoms, potential matches in the retrieved info, and severity.\n"
        "3. Assess Severity: If the user's description or the context's severity warnings strongly suggest a serious condition, state this in 'SEVERITY_ASSESSMENT:' and set 'ACTION: suggest_hospital'.\n"
        "4. Check Information Sufficiency: If the information is insufficient for a suggestion (e.g., symptoms are too vague, context doesn't match well, crucial details missing), explain why in 'REASONING:', formulate a single, specific clarifying question under 'NEXT_QUESTION:', and set 'ACTION: ask_question'.\n"
        "5. Provide Suggestion (If Safe and Sufficient): If severity is low and information is sufficient, state the likely condition under 'DIAGNOSIS:' and suggest 3 relevant non-prescription options (name, dosage, warnings) mentioned in the retrieved context under 'MEDICATION_SUGGESTION:'. Set 'ACTION: provide_info'. Do NOT invent medications or dosages.\n"
        "6. ALWAYS include a disclaimer that you are an AI and cannot provide medical advice.\n\n"
        "EXAMPLE (Suggestion):\n"
        "REASONING: User reports sneezing and itchy eyes. Retrieved context for Allergic Rhinitis matches these symptoms. Severity indicators don't seem triggered. Loratadine is listed as a non-drowsy option in the context.\n"
        "DIAGNOSIS: Based on the symptoms and context, it might be Allergic Rhinitis.\n"
        "MEDICATION_SUGGESTION: The context mentions Loratadine (10mg once daily) as a possible OTC option for relief. Refer to product packaging for full details. Disclaimer: I am an AI assistant... consult a healthcare professional.\n"
        "ACTION: provide_info\n\n"
        "EXAMPLE (Needs Clarification):\n"
        "REASONING: User reports a cough. Retrieved context mentions Common Cold and Acute Bronchitis. Need to know if the cough produces mucus to differentiate based on context.\n"
        "NEXT_QUESTION: Are you coughing up any mucus or phlegm?\n"
        "ACTION: ask_question\n\n"
        "EXAMPLE (Severe):\n"
        "REASONING: User mentions severe chest pain. This matches severity warnings in multiple contexts and is a critical symptom.\n"
        "SEVERITY_ASSESSMENT: Severe chest pain requires immediate medical evaluation.\n"
        "ACTION: suggest_hospital\n\n"
        "Now, analyze the following:"
    )

    messages = [{"role": "system", "content": system_message}]

    history_limit = 6 
    for turn in history[-history_limit:]:
        if isinstance(turn, dict) and turn.get('role') in ['user', 'assistant'] and isinstance(turn.get('content'), str):
            role = turn['role'] if turn['role'] == 'user' else 'assistant'
            messages.append({"role": role, "content": turn['content']})

    # Construct the user part of the prompt including context
    user_task_prompt = (
        f"Conversation History (if any) is above.\n\n"
        f"Current User Input: {user_input}\n\n"
        f"--- Start Retrieved Information ---\n{retrieved_context_str}\n--- End Retrieved Information ---\n\n"
        f"[Instruction] Follow the Chain-of-Thought process outlined in the system message based *only* on the provided input and retrieved information. Provide your response using the specified markers (REASONING:, SEVERITY_ASSESSMENT:, DIAGNOSIS:, MEDICATION_SUGGESTION:, NEXT_QUESTION:, ACTION:) and definitely give suggestions from the retrived context."
    )
    messages.append({"role": "user", "content": user_task_prompt})

    # LLM Generation
    logger.debug("Reason API: Sending messages to LLM service...")
    llm_response_data = llm_service.generate(messages)
    # Check if LLM service returned an error
    if "error" in llm_response_data:
        logger.error(f"Reason API: LLM generation failed: {llm_response_data['error']}")
        return jsonify({"error": "AI service failed", "details": llm_response_data["error"]}), 503

    # llm_response_data now contains the parsed fields like 'reasoning', 'action', 'response_text' etc.
    determined_action = llm_response_data.get("action", "provide_info")

    explanation_data = {"status": "not_generated", "message": "XAI explanation not generated by this endpoint."}
    if generate_explanation_flag:
        logger.info("Reason API: generate_explanation flag is true, but SHAP is generated on demand via /api/explain.")

    # Construct the final JSON response to send back to Node.js
    final_response = {
        "ai_response": llm_response_data,
        "explanation": explanation_data,
        "session_id": session_id
    }

    logger.info(f"Reason API: Successfully processed request for session {session_id}. Determined Action: {determined_action}.")
    return jsonify(final_response), 200