import logging
import re
import os

# Attempt to import vector_store_service for Neo4j access
vector_store_service = None
kg_service_available = False
try:
    from ..services import vector_store_service
    kg_service_available = True
    logger_verifier = logging.getLogger(__name__)
    logger_verifier.debug("Successfully imported vector_store_service.")
except ImportError:
    logger_verifier = logging.getLogger(__name__)
    if not logger_verifier.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger_verifier.warning("vector_store_service import failed. KG live checks will be disabled if driver not passed.")

# Action keyword heuristics (used in action consistency checks)
FIND_HOSPITAL_KEYWORDS = ["hospital", "urgent care", "emergency", "severe", "critical", "see a doctor immediately", "er"]
FIND_PHARMACY_KEYWORDS = ["pharmacy", "medication", "prescription", "drug store", "over-the-counter", "take", "dosage"]

# --- Neo4j Transaction Helper ---
def _check_contraindication_tx(tx, medication_name: str, patient_factors: list) -> bool:
    """Query Neo4j to check if medication is contraindicated for given factors."""
    if not medication_name or not patient_factors:
        return False

    med_name_lower = medication_name.lower()
    factors_lower = [str(f).lower() for f in patient_factors if isinstance(f, str) and f]
    if not factors_lower:
        return False

    query = """
    MATCH (m:Medication)
    WHERE toLower(m.name) CONTAINS $med_name_lower
    MATCH (m)-[:HAS_CONTRAINDICATION]->(c:Condition)
    WHERE toLower(c.name) IN $factors_lower
    WITH count(c) AS contraindication_count
    RETURN contraindication_count > 0 AS isContraindicated
    """
    try:
        result = tx.run(query, med_name_lower=med_name_lower, factors_lower=factors_lower)
        record = result.single()
        return record["isContraindicated"] if record else False
    except Exception as e:
        logger_verifier.error(f"Neo4j error checking '{medication_name}': {e}", exc_info=True)
        raise

# --- Verifiers ---

def verify_contraindication_live(generated_response: str, patient_profile: dict, **kwargs) -> float:
    """Checks Neo4j live KG for contraindications in suggested medication."""
    driver = vector_store_service.get_driver() if kg_service_available else kwargs.get('neo4j_driver')
    if driver is None:
        logger_verifier.warning("Neo4j driver unavailable. Returning neutral score (0.5).")
        return 0.5

    med_match = re.search(r"MEDICATION_SUGGESTION:\s*([^\n]+)", generated_response, re.IGNORECASE)
    if not med_match:
        return 1.0

    recommended_med = med_match.group(1).split(',')[0].split(' or ')[0].strip()
    if not recommended_med:
        return 1.0

    allergies = patient_profile.get("allergies", [])
    conditions = patient_profile.get("conditions", [])
    patient_factors = list(set(
        [str(f).lower() for f in allergies + conditions if isinstance(f, str) and f]
    ))

    if not patient_factors:
        return 1.0

    try:
        with driver.session(database="neo4j") as session:
            is_contraindicated = session.execute_read(
                _check_contraindication_tx, recommended_med, patient_factors
            )
        return 0.0 if is_contraindicated else 1.0
    except Exception as e:
        logger_verifier.error(f"Neo4j verification failed: {e}", exc_info=True)
        return 0.5

def verify_contraindication_sim(generated_response: str, patient_profile: dict, **kwargs) -> float:
    """Checks simulated KG (in-memory) for contraindications."""
    kg_sim = kwargs.get('kg_data_sim')
    if kg_sim is None:
        return 0.5

    med_match = re.search(r"MEDICATION_SUGGESTION:\s*([^\n]+)", generated_response, re.IGNORECASE)
    if not med_match:
        return 1.0

    recommended_med = med_match.group(1).split(',')[0].split(' or ')[0].strip()
    if not recommended_med:
        return 1.0

    med_key = next((k for k in kg_sim if k.lower() == recommended_med.lower()), None)
    if not med_key:
        return 1.0

    patient_factors = set(
        str(f).lower() for f in patient_profile.get("allergies", []) + patient_profile.get("conditions", []) if isinstance(f, str)
    )
    if not patient_factors:
        return 1.0

    sim_contras = set(c.lower() for c in kg_sim.get(med_key, {}).get("contraindications", []))
    return 0.0 if not patient_factors.isdisjoint(sim_contras) else 1.0

def verify_format(generated_response: str, **kwargs) -> float:
    """Checks that the response contains required structural markers."""
    text_upper = generated_response.upper()
    reasoning_ok = "REASONING:" in text_upper
    action_marker_ok = "ACTION:" in text_upper
    content_ok = any(marker in text_upper for marker in ["NEXT_QUESTION:", "DIAGNOSIS:", "MEDICATION_SUGGESTION:"])

    action_valid = False
    if action_marker_ok:
        action_part = generated_response[text_upper.find("ACTION:") + len("ACTION:"):].strip().lower().replace("_", "")
        action_valid = action_part in {"askquestion", "findpharmacy", "findhospital", "provideinfo"}

    if reasoning_ok and action_valid and content_ok:
        return 1.0

    logger_verifier.debug(f"Format verifier failed: R:{reasoning_ok}, A:{action_valid}, C:{content_ok}")
    return 0.0

def verify_action_consistency(generated_response: str, **kwargs) -> float:
    """Checks if the declared ACTION matches the content provided."""
    text = generated_response.lower()
    action_match = re.search(r"action:\s*(\S+)", text)
    action = action_match.group(1).replace("_", "") if action_match else None

    med_present = "medication_suggestion:" in text
    question_present = "next_question:" in text
    hospital_present = any(k in text for k in FIND_HOSPITAL_KEYWORDS)

    if action == "findpharmacy" and not med_present:
        return 0.5
    if action == "findhospital" and not hospital_present:
        return 0.5
    if action == "askquestion" and not question_present:
        return 0.5
    if action == "provideinfo" and (med_present or question_present):
        return 0.5
    if med_present and action != "findpharmacy":
        return 0.5

    return 1.0

VERIFIER_FUNCTIONS = [
    verify_contraindication_live,
    verify_contraindication_sim,
    verify_format,
    verify_action_consistency,
]

logger_verifier.info(f"Loaded {len(VERIFIER_FUNCTIONS)} verifier functions.")
