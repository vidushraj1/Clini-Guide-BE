import logging
import requests
import os
import torch
import time
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# Flexible import for app vs standalone usage
try:
    from ..config import Config
    from .train_utils import setup_logging, check_rl_config_vars
    from .verifiers import VERIFIER_FUNCTIONS, verify_contraindication_live, verify_contraindication_sim, verify_format
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from app.config import Config
        from app.training.train_utils import setup_logging, check_rl_config_vars
        from app.training.verifiers import VERIFIER_FUNCTIONS, verify_contraindication_live, verify_contraindication_sim, verify_format
    except ImportError as e:
        print(f"ERROR: Standalone import failed: {e}")
        import logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        def setup_logging(): return logging.getLogger(__name__)
        def check_rl_config_vars(): return True
        VERIFIER_FUNCTIONS = []
        class Config: pass

logger = setup_logging()
MIN_FEEDBACK_SAMPLES = 5

def fetch_feedback_data():
    """Fetches feedback entries from the Node backend."""
    feedback_endpoint = Config.FEEDBACK_API_ENDPOINT
    logger.info(f"Fetching feedback from: {feedback_endpoint}")
    headers = {}
    if getattr(Config, 'NODE_API_KEY', None):
        headers["Authorization"] = f"Bearer {Config.NODE_API_KEY}"
    try:
        response = requests.get(feedback_endpoint, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        if (isinstance(data, dict) and data.get('success') and 
            isinstance(data.get('data', {}).get('feedbackEntries'), list)):
            entries = data['data']['feedbackEntries']
            logger.info(f"Fetched {len(entries)} feedback entries.")
            return entries
        else:
            logger.error(f"Unexpected feedback structure: {json.dumps(data)[:200]}...")
    except Exception as e:
        logger.error(f"Error fetching feedback: {e}", exc_info=True)
    return None

def create_prompt_from_feedback_colab(feedback_entry, tokenizer, max_length=512):
    """Constructs formatted prompt from a feedback entry."""
    history = feedback_entry.get("history", [])
    profile = feedback_entry.get("profile", {})
    profile_parts = []
    if profile.get('age'): profile_parts.append(f"Age {profile['age']}")
    if profile.get('gender'): profile_parts.append(profile['gender'])
    if profile.get('allergies'): profile_parts.append(f"Allergies: {', '.join(profile['allergies'][:2])}...")
    if profile.get('existingMedications'): profile_parts.append(f"Taking: {', '.join(profile['existingMedications'][:2])}...")
    profile_str = f" [Patient: {', '.join(profile_parts)}]" if profile_parts else ""

    system_prompt = (
        f"You are a helpful medical AI.{profile_str} Respond using markers "
        f"(REASONING:, NEXT_QUESTION:, DIAGNOSIS:, MEDICATION_SUGGESTION:, ACTION:)."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-4:]:
        if isinstance(turn, dict) and turn.get('role') and turn.get('content'):
            messages.append(turn)

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logger.error(f"Tokenizer template error: {e}")
        return None

def preprocess_feedback_for_grpo_colab(feedback_data, tokenizer, max_prompt_length):
    """Prepares prompt+feedback examples for GRPO training."""
    processed = []
    logger.info(f"Processing {len(feedback_data)} feedback entries...")
    for i, entry in enumerate(feedback_data):
        feedback = entry.get("feedback", {})
        if 'thumbs' not in feedback:
            continue
        prompt = create_prompt_from_feedback_colab(entry, tokenizer, max_prompt_length)
        if not prompt:
            continue
        processed.append({
            "prompt": prompt,
            "feedback_details": feedback,
            "patient_profile": entry.get("profile", {})
        })
    logger.info(f"Preprocessed {len(processed)} valid feedback entries.")
    return processed

def doctor_feedback_reward(completions, **kwargs) -> list:
    """Simple reward based on thumbs up/down from human."""
    thumbs = kwargs.get("feedback_details", {}).get("thumbs")
    if thumbs == 'up':
        return [1.0] * len(completions)
    elif thumbs == 'down':
        return [0.0] * len(completions)
    return [0.5] * len(completions)

def combined_reward_function(completions: list, **kwargs) -> list:
    """Blends human feedback + verifier-based reward."""
    num = len(completions)
    if not num: return []

    rlhf = torch.tensor(doctor_feedback_reward(completions, **kwargs), dtype=torch.float32)
    rlvr = torch.zeros(num)
    penalty = getattr(Config, 'VERIFIER_PENALTY', -5.0)

    profile = kwargs.get("patient_profile", {})
    for i, output in enumerate(completions):
        if not output:
            rlvr[i] = 0.0
            continue
        scores = []
        critical = False
        for idx, verifier in enumerate(VERIFIER_FUNCTIONS):
            try:
                score = verifier(generated_response=output, patient_profile=profile, **kwargs)
                scores.append(score)
                if idx == 0 and score == 0.0:
                    critical = True
            except Exception as e:
                logger.error(f"Verifier '{verifier.__name__}' error: {e}")
                scores.append(0.5)
        if scores:
            rlvr[i] = sum(scores) / len(scores)
            if critical:
                rlvr[i] += penalty
        else:
            rlvr[i] = 0.5

    total = (
        getattr(Config, 'RLHF_WEIGHT', 0.6) * rlhf +
        getattr(Config, 'RLVR_WEIGHT', 0.4) * rlvr
    )
    return torch.clamp(total, min=-10.0, max=10.0).tolist()

def run_combined_rl_cycle():
    """Runs one training cycle of combined RLHF + RLVR with GRPO."""
    logger.info("--- Starting Combined RL Cycle ---")
    start = time.time()
    if not check_rl_config_vars():
        return

    # Attempt to initialize Neo4j if available
    neo4j_driver = None
    use_kg = False
    try:
        from ..services import vector_store_service
        vector_store_service.init_driver(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        if vector_store_service.verify_connection():
            neo4j_driver = vector_store_service.get_driver()
            use_kg = True
            logger.info("Neo4j connected.")
    except Exception as e:
        logger.warning(f"Neo4j unavailable: {e}")

    # Load model & tokenizer
    logger.info(f"Loading model: {Config.LLM_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        Config.LLM_MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Load or apply LoRA adapters
    adapter_path = Config.STUDENT_MODEL_ADAPTERS_PATH
    adapter_file = os.path.join(adapter_path, 'adapter_config.json')
    if adapter_path and os.path.exists(adapter_file):
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Load and prepare feedback data
    feedback = fetch_feedback_data()
    if not feedback or len(feedback) < MIN_FEEDBACK_SAMPLES:
        logger.warning("Not enough feedback for training.")
        return

    train_data = preprocess_feedback_for_grpo_colab(feedback, tokenizer, Config.GRPO_MAX_PROMPT)
    if not train_data:
        logger.error("No trainable data generated.")
        return

    dataset = Dataset.from_list(train_data)

    # Setup training configuration
    output_dir = os.path.join(Config.TRAINING_OUTPUT_DIR, f"grpo_run_{int(time.time())}")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=Config.GRPO_LR,
        gradient_accumulation_steps=Config.GRPO_ACCUM_STEPS,
        num_train_epochs=Config.GRPO_EPOCHS,
        max_completion_length=Config.GRPO_MAX_COMPLETION,
        num_generations=Config.GRPO_NUM_GENERATIONS,
        max_prompt_length=Config.GRPO_MAX_PROMPT,
        beta=Config.GRPO_BETA,
        per_device_train_batch_size=Config.GRPO_BATCH_SIZE,
        remove_unused_columns=False,
        logging_steps=max(1, len(dataset) // (Config.GRPO_BATCH_SIZE * Config.GRPO_ACCUM_STEPS * 10)),
        save_strategy="epoch",
        report_to=["tensorboard"],
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=42,
        warmup_ratio=0.1,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        push_to_hub=bool(Config.HF_HUB_REPO_ID),
        hub_model_id=Config.HF_HUB_REPO_ID,
        hub_token=Config.HF_HUB_TOKEN,
        hub_private_repo=True
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_func=combined_reward_function,
        num_reward_funcs=1
    )

    logger.info(f"Starting GRPO training for {grpo_args.num_train_epochs} epochs...")
    trainer.train()
    logger.info("Training complete. Saving model...")

    os.makedirs(adapter_path, exist_ok=True)
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    if grpo_args.push_to_hub:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=grpo_args.hub_model_id, exist_ok=True, private=grpo_args.hub_private_repo)
            trainer.push_to_hub(commit_message=f"RLHF+RLVR update {int(time.time())}")
        except Exception as e:
            logger.error(f"Push to Hub failed: {e}", exc_info=True)

    if use_kg and vector_store_service:
        vector_store_service.close_driver()

    logger.info(f"--- RL cycle finished in {time.time() - start:.2f}s ---")

if __name__ == "__main__":
    logger.info("Launching manual run of RL training cycle.")
    if check_rl_config_vars():
        run_combined_rl_cycle()
    else:
        logger.error("Configuration invalid. Exiting.")
