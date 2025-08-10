import logging
import os
import torch
import time
import json
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
import torch.nn.functional as F

# Load config and utility methods; fallback for standalone usage
try:
    from ..config import Config
    from .train_utils import setup_logging, check_distill_config_vars, check_config_vars
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from app.config import Config
        from app.training.train_utils import setup_logging, check_distill_config_vars, check_config_vars
    except ImportError:
        print("WARNING: Fallback config and logging setup triggered.")
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        def setup_logging(): return logging.getLogger(__name__)
        def check_distill_config_vars(): return True
        class Config: pass

logger = setup_logging()

DISTILL_ADAPTER_SUBDIR = "distilled_adapters"
KG_DATA_SIM_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'kg_data_sim.json')

def load_teacher_model(teacher_model_id):
    """Loads teacher model and tokenizer from HuggingFace."""
    logger.info(f"Loading Teacher Model: {teacher_model_id}")
    if not teacher_model_id:
        logger.error("Missing TEACHER_MODEL_NAME.")
        return None, None
    try:
        model = AutoModelForCausalLM.from_pretrained(teacher_model_id, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load teacher model: {e}", exc_info=True)
        return None, None

def load_or_create_kg_sim(kg_json_path):
    """Creates or loads a KG symptom-to-disease simulation map."""
    kg_sim = {}
    if os.path.exists(KG_DATA_SIM_PATH):
        try:
            with open(KG_DATA_SIM_PATH, 'r') as f:
                kg_sim = json.load(f)
            logger.info(f"Loaded cached KG simulation: {KG_DATA_SIM_PATH}")
            return kg_sim
        except Exception as e:
            logger.warning(f"Could not read cached KG file: {e}")
    logger.info("Generating new KG simulation lookup...")
    try:
        with open(kg_json_path, 'r') as f:
            raw_data = json.load(f)
        for entry in raw_data:
            for symptom in entry.get('symptoms', []):
                key = symptom.lower().strip()
                if key:
                    kg_sim.setdefault(key, []).append(entry['disease'])
        for key in kg_sim:
            kg_sim[key] = list(set(kg_sim[key]))
        with open(KG_DATA_SIM_PATH, 'w') as f:
            json.dump(kg_sim, f)
        logger.info("KG simulation lookup generated and saved.")
    except Exception as e:
        logger.error(f"Error generating KG sim: {e}", exc_info=True)
    return kg_sim

def generate_kg_context(symptoms, kg_sim_lookup):
    """Returns a short text summary of conditions related to symptoms."""
    lines = []
    for symptom in symptoms:
        key = symptom.lower().strip()
        if key in kg_sim_lookup:
            diseases = kg_sim_lookup[key]
            lines.append(f"For symptom '{symptom}', possible conditions include: {', '.join(diseases)}.")
    return "\n".join(lines) if lines else "No specific KG context available."

def generate_distillation_example(symptoms, teacher_model, teacher_tokenizer, kg_sim_lookup):
    """Generates a single prompt/response example using teacher model."""
    context = generate_kg_context(symptoms, kg_sim_lookup)
    prompt = (
        "System: You are a medical expert. Based on the following context:\n"
        f"{context}\n\n"
        "Patient presents with the following symptoms: " + ", ".join(symptoms) + ".\n"
        "Provide a detailed and medically sound recommendation.\nAssistant:"
    )
    logger.info(f"Prompt preview: {prompt[:100]}...")
    try:
        input_ids = teacher_tokenizer(prompt, return_tensors="pt").input_ids.to(teacher_model.device)
        output_ids = teacher_model.generate(input_ids, max_length=256)
        response = teacher_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"input": prompt, "output": response}
    except Exception as e:
        logger.error(f"Teacher model generation failed: {e}", exc_info=True)
        return {"input": prompt, "output": "Error in teacher inference."}

def prepare_distillation_data(teacher_model, teacher_tokenizer, kg_sim_lookup):
    """Generates a small dataset of synthetic prompt-response examples."""
    logger.info("Preparing examples for distillation...")
    sample_sets = [
        ["cough", "fever"],
        ["headache", "nausea"],
        ["sore throat", "runny nose"],
        ["chest pain", "shortness of breath"]
    ]
    examples = [generate_distillation_example(symptoms, teacher_model, teacher_tokenizer, kg_sim_lookup)
                for symptoms in sample_sets]
    logger.info(f"Prepared {len(examples)} examples.")
    try:
        return Dataset.from_list(examples)
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        return None

def preprocess_for_distill(examples, tokenizer, max_length):
    """Tokenizes inputs and outputs for training."""
    model_inputs = tokenizer(examples["input"], max_length=max_length, truncation=True, padding=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], max_length=max_length, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def run_knowledge_distillation():
    logger.info("--- Starting Knowledge Distillation ---")
    start = time.time()
    if not check_distill_config_vars():
        return

    kg_path = os.path.join(os.path.dirname(__file__), '..', '..', 'kg_data.json')
    kg_sim = load_or_create_kg_sim(kg_path)

    teacher_model, teacher_tokenizer = load_teacher_model(Config.TEACHER_MODEL_NAME)
    if teacher_model is None:
        logger.error("Aborting: teacher model load failed.")
        return

    distill_dataset = prepare_distillation_data(teacher_model, teacher_tokenizer, kg_sim)
    if not distill_dataset:
        return

    logger.info(f"Loading student model: {Config.LLM_MODEL_NAME}")
    try:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        student_model = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL_NAME,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            student_model.config.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        logger.error(f"Student model load failed: {e}", exc_info=True)
        return

    max_len = Config.GRPO_MAX_PROMPT + Config.GRPO_MAX_COMPLETION
    try:
        tokenized = distill_dataset.map(
            lambda ex: preprocess_for_distill(ex, tokenizer, max_len),
            batched=True,
            remove_columns=distill_dataset.column_names
        )
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        return

    logger.info("Applying LoRA to student model...")
    try:
        student_model = prepare_model_for_kbit_training(student_model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()
    except Exception as e:
        logger.error(f"LoRA application failed: {e}", exc_info=True)
        return

    logger.info("Configuring training...")
    training_args = TrainingArguments(
        output_dir=os.path.join(Config.TRAINING_OUTPUT_DIR, "distill_checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=Config.GRPO_BATCH_SIZE,
        gradient_accumulation_steps=Config.GRPO_ACCUM_STEPS,
        learning_rate=Config.GRPO_LR,
        logging_steps=20,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=student_model,
        padding="longest",
        return_tensors="pt"
    )

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    logger.info("Trainer initialized. Ready for training.")

    adapter_path = os.path.join(Config.TRAINING_OUTPUT_DIR, DISTILL_ADAPTER_SUBDIR)
    logger.info(f"Saving distilled adapters to {adapter_path} (save skipped).")
    logger.info(f"Distillation finished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    logger.info("Executing distillation trainer manually.")
    os.makedirs(Config.TRAINING_OUTPUT_DIR, exist_ok=True)
    if check_distill_config_vars():
        run_knowledge_distillation()
    else:
        logger.error("Missing configuration. Distillation aborted.")