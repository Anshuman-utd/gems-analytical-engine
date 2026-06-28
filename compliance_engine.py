# At the top of compliance_engine.py — replace anthropic import with:
import os
import json
import logging
from groq import Groq
from pydantic import BaseModel, ValidationError, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TenderRequirements(BaseModel):
    min_turnover_lakhs: float        # minimum annual turnover in lakhs
    min_experience_years: int        # years of prior experience required
    mse_preferred: bool              # does this tender give MSE purchase preference?
    emd_required_lakhs: float        # Earnest Money Deposit amount
    delivery_days: int               # delivery period in days
    product_specs: list[str]         # key technical specifications

    @field_validator("min_turnover_lakhs")
    @classmethod
    def turnover_must_be_positive(cls, v):
        # Validator runs automatically when Pydantic parses the LLM response.
        # If LLM hallucinates a negative turnover, this catches it.
        if v < 0:
            raise ValueError("Turnover cannot be negative")
        return v

    @field_validator("min_experience_years")
    @classmethod
    def experience_must_be_reasonable(cls, v):
        # Government tenders rarely ask for >20 years. Flag if LLM hallucinates.
        if v < 0 or v > 20:
            raise ValueError(f"Experience years {v} is outside reasonable range 0-20")
        return v


# ─────────────────────────────────────────────────────────────
# SECTION 2: VENDOR PROFILES
# These are programmatically generated — not from any external source.
# Realistic variances that test different compliance scenarios.
# ─────────────────────────────────────────────────────────────

VENDOR_PROFILES = [
    {
        "name": "Vendor A — TechSupply India Pvt Ltd",
        "turnover_lakhs": 100.0,    # ₹1 Crore
        "experience_years": 5,
        "is_mse": True,             # Micro & Small Enterprise — gets purchase preference
        "is_mii": True,             # Make in India — additional preference
        "emd_capacity_lakhs": 5.0,
    },
    {
        "name": "Vendor B — GlobalIT Solutions",
        "turnover_lakhs": 30.0,     # ₹30 Lakhs — likely to fail turnover threshold
        "experience_years": 2,
        "is_mse": False,
        "is_mii": False,
        "emd_capacity_lakhs": 2.0,
    },
    {
        "name": "Vendor C — BharatTech Systems",
        "turnover_lakhs": 250.0,    # ₹2.5 Crore — comfortably eligible
        "experience_years": 8,
        "is_mse": True,
        "is_mii": False,
        "emd_capacity_lakhs": 10.0,
    },
]


# ─────────────────────────────────────────────────────────────
# SECTION 3: LLM EXTRACTION — with hardening
# The LLM's ONLY job is to read messy PDF text and return JSON.
# We give it an exact schema to follow via the system prompt.
# ─────────────────────────────────────────────────────────────

FALLBACK_REQUIREMENTS = TenderRequirements(
    min_turnover_lakhs=50.0,
    min_experience_years=3,
    mse_preferred=True,
    emd_required_lakhs=1.0,
    delivery_days=30,
    product_specs=["FALLBACK: Could not parse tender — manual review required"]
)

def extract_requirements_with_llm(tender_text: str) -> TenderRequirements:
    """
    Uses Groq's API (free, fast) to parse tender text into structured JSON.
    We use llama3-8b — small enough to be fast, capable enough for extraction.

    The hardening logic is identical regardless of which LLM we use:
    Groq returns text → we parse JSON → Pydantic validates → fallback if anything fails.
    Swapping Groq for Anthropic is a 3-line change — that's good architecture.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.error("GROQ_API_KEY not configured")
        return FALLBACK_REQUIREMENTS

    client = Groq(api_key=api_key)

    system_prompt = """
You are an information extraction system for Government e-Marketplace (GeM) tender documents.

Your ONLY task is to extract eligibility and procurement requirements that are EXPLICITLY mentioned in the tender.

Rules:
- Return ONLY a valid JSON object.
- Do NOT include markdown.
- Do NOT include ```json or backticks.
- Do NOT include explanations or extra text.
- Do NOT infer missing values.
- Do NOT estimate values.
- Do NOT use prior knowledge.
- Extract ONLY information explicitly present in the tender.
- If a value is not mentioned, return the default specified below.
- Product specifications should contain at most 5 important technical requirements.

Return this exact JSON schema:

{
  "min_turnover_lakhs": <float, annual turnover in lakhs INR, use 0 if not mentioned>,
  "min_experience_years": <int, years of experience, use 0 if not mentioned>,
  "mse_preferred": <boolean, true only if MSE purchase preference is explicitly mentioned>,
  "emd_required_lakhs": <float, EMD/Bid Security in lakhs, use 0 if not mentioned>,
  "delivery_days": <int, delivery period in days, use 30 if not mentioned>,
  "product_specs": [
    "<technical specification 1>",
    "<technical specification 2>"
  ]
}
"""

    logger.info("Calling Groq LLM for requirement extraction...")

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",   # free tier, fast, sufficient for extraction
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Extract requirements:\n\n{tender_text[:6000]}"}
            ],
            temperature=0,            # temperature=0 → deterministic output, no creativity
            max_tokens=600,
        )

        raw = response.choices[0].message.content.strip()
        logger.info(f"Groq response (first 200 chars): {raw[:200]}")

        # Remove markdown fences if model adds them despite instructions
        # Some models return ```json ... ``` even when told not to
        raw = raw.replace("```json", "").replace("```", "").strip()

        parsed_dict  = json.loads(raw)
        requirements = TenderRequirements(**parsed_dict)
        logger.info("Extraction successful and Pydantic-validated")
        return requirements

    except json.JSONDecodeError as e:
        logger.error(f"Groq returned non-JSON: {e}")
        logger.warning("Using fallback requirements")
        return FALLBACK_REQUIREMENTS

    except ValidationError as e:
        logger.error(f"Pydantic validation failed: {e}")
        logger.warning("Using fallback requirements")
        return FALLBACK_REQUIREMENTS

    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        logger.warning("Using fallback requirements")
        return FALLBACK_REQUIREMENTS

def evaluate_vendor(vendor: dict, requirements: TenderRequirements) -> dict:
    """
    Evaluates a single vendor against extracted tender requirements.
    Every check is a simple boolean comparison — fully auditable.
    Returns a structured report that the UI can display directly.
    """
    checks = {}
    failures = []

    # Check 1: Annual turnover
    turnover_pass = vendor["turnover_lakhs"] >= requirements.min_turnover_lakhs
    checks["turnover"] = {
        "required": f"₹{requirements.min_turnover_lakhs}L",
        "vendor_has": f"₹{vendor['turnover_lakhs']}L",
        "pass": turnover_pass
    }
    if not turnover_pass:
        failures.append(
            f"Turnover ₹{vendor['turnover_lakhs']}L below required ₹{requirements.min_turnover_lakhs}L"
        )

    # Check 2: Experience
    exp_pass = vendor["experience_years"] >= requirements.min_experience_years
    checks["experience"] = {
        "required": f"{requirements.min_experience_years} years",
        "vendor_has": f"{vendor['experience_years']} years",
        "pass": exp_pass
    }
    if not exp_pass:
        failures.append(
            f"Experience {vendor['experience_years']}yr below required {requirements.min_experience_years}yr"
        )

    # Check 3: EMD capacity
    emd_pass = vendor["emd_capacity_lakhs"] >= requirements.emd_required_lakhs
    checks["emd"] = {
        "required": f"₹{requirements.emd_required_lakhs}L",
        "vendor_has": f"₹{vendor['emd_capacity_lakhs']}L",
        "pass": emd_pass
    }
    if not emd_pass:
        failures.append(
            f"EMD capacity ₹{vendor['emd_capacity_lakhs']}L below required ₹{requirements.emd_required_lakhs}L"
        )

    # Check 4: MSE Purchase Preference bonus
    # GeM policy: MSE vendors get a 15% price preference margin.
    # This doesn't affect compliance but feeds into win probability.
    mse_advantage = vendor["is_mse"] and requirements.mse_preferred
    checks["mse_preference"] = {
        "applicable": mse_advantage,
        "note": "15% purchase preference margin applies" if mse_advantage else "No MSE preference"
    }

    is_eligible = len(failures) == 0

    return {
        "vendor_name": vendor["name"],
        "is_eligible": is_eligible,
        "checks": checks,
        "failures": failures,
        "mse_advantage": mse_advantage,
        "summary": "ELIGIBLE ✓" if is_eligible else f"INELIGIBLE ✗ — {'; '.join(failures)}"
    }


def run_compliance_check(tender_text: str) -> list[dict]:
    """
    Full pipeline: extract requirements → evaluate all vendors.
    Returns list of evaluation reports, one per vendor.
    """
    requirements = extract_requirements_with_llm(tender_text)

    logger.info(f"Extracted requirements: {requirements.model_dump()}")

    results = []
    for vendor in VENDOR_PROFILES:
        result = evaluate_vendor(vendor, requirements)
        results.append(result)
        logger.info(f"{vendor['name']}: {result['summary']}")

    return results, requirements