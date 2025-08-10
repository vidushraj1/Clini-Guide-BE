import logging
import sys
import os

# Try importing config; fallback for standalone runs (e.g., `python -m app.training.train_utils`)
try:
    from ..config import Config
except ImportError:
    print("Warning: Relative import of Config failed in train_utils. Attempting direct import.")
    try:
        from app.config import Config
    except ImportError:
        print("FATAL: Cannot import Config in train_utils. Ensure PYTHONPATH is set or run from project root.")
        logging.basicConfig(level=logging.ERROR)
        logging.critical("Config class could not be imported.")
        class Config: pass  # Dummy fallback

# --- Logging Setup ---
def setup_logging():
    """Configures logging behavior for training scripts."""
    try:
        env = os.environ.get('LOG_LEVEL', os.environ.get('FLASK_ENV', 'INFO')).upper()
        if env == 'DEVELOPMENT':
            log_level_name = 'DEBUG'
        elif env == 'PRODUCTION':
            log_level_name = 'INFO'
        elif env in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            log_level_name = env
        else:
            print(f"Warning: Unknown LOG_LEVEL/FLASK_ENV '{env}'. Using INFO.")
            log_level_name = 'INFO'

        log_level = getattr(logging, log_level_name, logging.INFO)
    except Exception as e:
        print(f"Warning: Couldn't determine log level: {e}. Defaulting to INFO.")
        log_level = logging.INFO
        log_level_name = 'INFO'

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(log_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Reduce noise from commonly verbose libraries
    noisy_libs = [
        "transformers.generation.utils", "transformers.modeling_utils",
        "datasets.builder", "huggingface_hub.file_download",
        "asyncio", "httpx", "urllib3", "PIL",
        "peft.utils.other", "trl.trainer.ppo_trainer", "trl.trainer.utils",
        "neo4j", "filelock"
    ]
    for lib in noisy_libs:
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(f"Training logging configured. Root level: {log_level_name}")
    return root_logger

logger = logging.getLogger(__name__)

# --- Config Validation ---
def check_config_vars(required_vars: list):
    """Ensure required config values exist and create any needed directories."""
    logger.debug(f"Checking config vars: {required_vars}")
    missing = []

    for var in required_vars:
        value = getattr(Config, var, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            # Skip warnings in certain known-safe cases
            is_local_neo4j = hasattr(Config, 'NEO4J_URI') and Config.NEO4J_URI.startswith('bolt://localhost')
            is_encrypted = hasattr(Config, 'NEO4J_URI') and any(Config.NEO4J_URI.startswith(p) for p in ['neo4j+s', 'bolt+s'])

            if var == 'NEO4J_PASSWORD' and is_local_neo4j and not is_encrypted:
                logger.warning(f"'{var}' missing, but okay for unencrypted local Neo4j.")
            elif var == 'HF_HUB_TOKEN' and not getattr(Config, 'HF_HUB_REPO_ID', None):
                logger.debug(f"'{var}' not required (Hub push disabled).")
            elif var == 'TEACHER_MODEL_NAME' and 'distill_trainer.py' not in str(sys.argv):
                logger.debug(f"'{var}' not needed (not distillation run).")
            else:
                missing.append(var)

    if missing:
        logger.critical(f"Missing required config vars: {', '.join(missing)}. Check .env or config.")
        return False

    # Ensure directories exist for key path variables
    dirs_to_check = ['TRAINING_OUTPUT_DIR', 'STUDENT_MODEL_ADAPTERS_PATH']
    try:
        for var in dirs_to_check:
            dir_path = getattr(Config, var, None)
            if isinstance(dir_path, str) and dir_path:
                path = os.path.dirname(dir_path) if '.' in os.path.basename(dir_path) else dir_path
                if path:
                    os.makedirs(path, exist_ok=True)
                    logger.debug(f"Ensured directory exists: {path}")
        return True
    except Exception as e:
        logger.error(f"Error ensuring directories exist: {e}", exc_info=True)
        return False

def check_rl_config_vars():
    """Validates config for RLHF training."""
    logger.info("Validating config for RLHF training...")
    required = [
        'LLM_MODEL_NAME', 'LLM_DEVICE', 'TRAINING_OUTPUT_DIR', 'STUDENT_MODEL_ADAPTERS_PATH',
        'GRPO_LR', 'GRPO_ACCUM_STEPS', 'GRPO_EPOCHS', 'GRPO_MAX_COMPLETION',
        'GRPO_NUM_GENERATIONS', 'GRPO_MAX_PROMPT', 'GRPO_BETA', 'GRPO_BATCH_SIZE',
        'FEEDBACK_API_ENDPOINT', 'NODE_API_KEY',
        'NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD'
    ]
    return check_config_vars(required)

def check_distill_config_vars():
    """Validates config for distillation training."""
    logger.info("Validating config for distillation training...")
    required = [
        'LLM_MODEL_NAME',
        'TEACHER_MODEL_NAME',
        'TRAINING_OUTPUT_DIR',
        'STUDENT_MODEL_ADAPTERS_PATH'
    ]
    if not getattr(Config, 'TEACHER_MODEL_NAME', None):
        logger.error("TEACHER_MODEL_NAME is required but not set.")
        return False
    return check_config_vars(required)