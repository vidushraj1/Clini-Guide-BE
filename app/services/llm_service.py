import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import logging
import os
import time
import re

logger = logging.getLogger(__name__)

tokenizer = None
model = None
base_model_name_loaded = None
adapter_id_loaded = None
device = None

def load_model(model_name: str, device_pref: str, adapter_id_or_path: str | None = None, force_reload: bool = False):
    """
    Loads the base LLM (without quantization) and optional PEFT adapters
    onto the specified device. For CPU, we do not use BitsAndBytes quantization.
    """
    global tokenizer, model, device, base_model_name_loaded, adapter_id_loaded

    # Check if already loaded correctly.
    if model is not None and model_name == base_model_name_loaded and adapter_id_or_path == adapter_id_loaded and not force_reload:
        logger.info("LLM model and specified adapters already loaded. Skipping reload.")
        return

    if force_reload:
        logger.info("Force reloading model and adapters...")
        _cleanup_model()
    elif model is not None:
        logger.info("Configuration changed. Reloading model/adapters...")
        _cleanup_model()

    # Determine device.
    device = 'cpu' if device_pref.lower() == 'cpu' or not torch.cuda.is_available() else 'cuda'
    if device_pref.lower() == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'

    logger.info(f"Attempting to load LLM '{model_name}' on device '{device}'...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            logger.info("Set tokenizer pad token and padding side to left.")

        # Load Base Model: no quantization if device is CPU
        logger.info("Loading base model without bitsandbytes quantization...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        base_model.to(device)
        logger.info(f"Base model '{model_name}' loaded onto {device}.")

        if tokenizer.pad_token_id is not None:
            base_model.config.pad_token_id = tokenizer.pad_token_id
        else:
            base_model.config.pad_token_id = tokenizer.eos_token_id

        # Load Adapters
        final_model = base_model 
        loaded_adapter_id = None
        if adapter_id_or_path and isinstance(adapter_id_or_path, str) and adapter_id_or_path.strip():
            logger.info(f"Attempting to load PEFT adapters from '{adapter_id_or_path}'...")
            try:
                peft_model = PeftModel.from_pretrained(base_model, adapter_id_or_path, is_trainable=False)
                final_model = peft_model
                loaded_adapter_id = adapter_id_or_path
                logger.info(f"Successfully loaded PEFT adapters from '{adapter_id_or_path}'")
            except Exception as peft_e:
                logger.error(f"Failed to load PEFT adapters from '{adapter_id_or_path}': {peft_e}. Using base model only.", exc_info=True)
                loaded_adapter_id = None
        else:
            logger.info(f"No valid adapter ID/path provided ('{adapter_id_or_path}'). Using base model only.")
            loaded_adapter_id = None

        model = final_model
        base_model_name_loaded = model_name
        adapter_id_loaded = loaded_adapter_id

        adapter_status = f"adapters from '{adapter_id_loaded}'" if adapter_id_loaded else "base model only"
        logger.info(f"LLM Service Ready: '{base_model_name_loaded}' ({adapter_status}) on device: {device}.")

    except Exception as e:
        logger.error(f"FATAL: Failed overall model loading for '{model_name}': {e}", exc_info=True)
        _cleanup_model()

def _cleanup_model():
    """Cleans up the model and tokenizer."""
    global tokenizer, model, base_model_name_loaded, adapter_id_loaded
    logger.debug("Cleaning up existing LLM model and tokenizer...")
    m, t = model, tokenizer
    model, tokenizer = None, None
    base_model_name_loaded, adapter_id_loaded = None, None
    del m
    del t

def is_model_loaded():
    """Checks if the model and tokenizer are loaded."""
    return model is not None and tokenizer is not None

def generate(prompt_messages: list, max_new_tokens=300, temperature=0.6, top_p=0.9):
    """Generates a response using the loaded model and prompts."""
    if not is_model_loaded():
        logger.error("LLM not available for generation.")
        return {"error": "LLM service not initialized or failed to load."}

    model.eval()
    try:
        formatted_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        logger.debug(f"Formatted Prompt for Generation:\n{formatted_prompt[-500:]}...")
        model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)
        generation_kwargs = dict(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        logger.info("Starting LLM generation on CPU (this may be slow)...")
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**generation_kwargs)
        end_time = time.time()
        logger.info(f"LLM Generation completed in {end_time - start_time:.2f} seconds (CPU).")
        output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        parsed_output = parse_llm_output_structured(response_text)
        return parsed_output
    except Exception as e:
        logger.error(f"Error during LLM generation (CPU): {e}", exc_info=True)
        return {"error": f"Failed to generate LLM response: {e}"}

def parse_llm_output_structured(text: str) -> dict:
    """
    Parses the LLM output for structured Chain-of-Thought markers.
    Expected markers: REASONING:, NEXT_QUESTION:, DIAGNOSIS:, MEDICATION_SUGGESTION:, ACTION:
    Also includes severity assessment.
    """
    output = {"full_text": text.strip()}
    # Define possible actions clearly
    POSSIBLE_ACTIONS = ["provide_info", "ask_question", "suggest_hospital", "find_pharmacy"]
    # Markers expected from the CoT prompt
    markers = ["REASONING:", "NEXT_QUESTION:", "DIAGNOSIS:", "MEDICATION_SUGGESTION:", "SEVERITY_ASSESSMENT:", "ACTION:"]
    content_map = {}
    current_marker = None
    buffer = []

    for line in text.split('\n'):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        found_marker = False
        for marker in markers:
            if stripped_line.upper().startswith(marker.upper()):
                if current_marker and buffer:
                    content_map[current_marker] = "\n".join(buffer).strip()

                current_marker = marker.upper()
                buffer = [stripped_line[len(marker):].strip()]
                found_marker = True
                break

        if not found_marker and current_marker:
            buffer.append(stripped_line)

    if current_marker and buffer:
        content_map[current_marker] = "\n".join(buffer).strip()

    # Populate the output dictionary using uppercase keys
    output["reasoning"] = content_map.get("REASONING:")
    output["next_question"] = content_map.get("NEXT_QUESTION:")
    output["diagnosis"] = content_map.get("DIAGNOSIS:")
    output["medication_suggestion"] = content_map.get("MEDICATION_SUGGESTION:")
    output["severity_assessment"] = content_map.get("SEVERITY_ASSESSMENT:") # Added

    # Parse the ACTION marker
    action_raw = content_map.get("ACTION:")
    matched_action = None
    if action_raw:
        action_clean = ''.join(filter(str.isalnum, action_raw.lower()))
        for pa in POSSIBLE_ACTIONS:
            if ''.join(filter(str.isalnum, pa.lower())) == action_clean:
                matched_action = pa
                break

    # Determine final action based on parsing or inference
    if matched_action:
        output["action"] = matched_action
        logger.debug(f"Parsed action: {matched_action}")
    elif action_raw:
        logger.warning(f"LLM generated unknown action: '{action_raw}'. Defaulting based on content.")
        if output.get("severity_assessment") and "hospital" in output["severity_assessment"].lower():
             output["action"] = "suggest_hospital"
        elif output.get("next_question"):
             output["action"] = "ask_question"
        else:
             output["action"] = "provide_info"
    else:
        logger.debug("No ACTION marker found. Inferring action based on content.")
        if output.get("severity_assessment") and "hospital" in output["severity_assessment"].lower():
             output["action"] = "suggest_hospital"
        elif output.get("next_question"):
             output["action"] = "ask_question"
        else:
             output["action"] = "provide_info"
        logger.debug(f"Inferred action: {output['action']}")


    response_parts = []
    if output.get("reasoning"):
        response_parts.append(f"Reasoning: {output['reasoning']}")
    if output.get("diagnosis"):
        response_parts.append(f"Possible Condition: {output['diagnosis']}")
    if output.get("medication_suggestion"):
        response_parts.append(f"Suggestion: {output['medication_suggestion']}")
        response_parts.append("\nDisclaimer: I am an AI assistant. This is not medical advice. Please consult a healthcare professional for diagnosis and treatment.")
    if output.get("severity_assessment"):
        response_parts.append(f"Severity Assessment: {output['severity_assessment']}")
    if output.get("next_question"):
        response_parts.append(f"Next Question: {output['next_question']}")
    if output["action"] == "suggest_hospital" and not output.get("severity_assessment"):
         response_parts.append("Based on the potential severity, please seek immediate medical attention at a nearby hospital or clinic.")

    if response_parts:
         output["response_text"] = "\n".join(response_parts).strip()
    elif output["full_text"]:
         logger.debug("LLM output has no structured markers; using full_text as response_text.")
         output["response_text"] = output["full_text"]
         if any(kw in output["full_text"].lower() for kw in ["medication", "dosage", "treatment", "diagnos"]):
              output["response_text"] += "\n\nDisclaimer: I am an AI assistant. This is not medical advice. Please consult a healthcare professional for diagnosis and treatment."
    else:
         # Fallback if everything is empty
         output["response_text"] = "I encountered an issue processing the response. Could you please rephrase?"
         if not output.get("action"):
              output["action"] = "ask_question"


    # Ensure essential fields exist, even if None
    for key in ["reasoning", "next_question", "diagnosis", "medication_suggestion", "severity_assessment", "action", "response_text"]:
        if key not in output:
            output[key] = None

    # Clean up response_text slightly
    if output["response_text"]:
        output["response_text"] = output["response_text"].replace("\n\n", "\n").strip()


    return output
