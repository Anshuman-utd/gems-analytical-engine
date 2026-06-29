# At the top of compliance_engine.py — replace anthropic import with:
import os
import json
import re
import logging
from groq import Groq
from pydantic import BaseModel, ValidationError, field_validator
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TenderRequirements(BaseModel):
    min_turnover_lakhs: float
    min_experience_years: int
    mse_preferred: bool
    emd_required_lakhs: float
    delivery_days: int

    make_in_india_required: bool = False
    startup_allowed: bool = False

    product_specs: list[str]       # key technical specifications

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

    @field_validator("emd_required_lakhs")
    @classmethod
    def emd_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("EMD cannot be negative")
        return v

    @field_validator("delivery_days")
    @classmethod
    def delivery_days_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Delivery days cannot be negative")
        return v
    
    @field_validator("product_specs")
    @classmethod
    def validate_product_specs(cls, specs):
        if len(specs) > 5:
            raise ValueError("Maximum 5 product specifications allowed")

        cleaned = []

        for spec in specs:
            spec = spec.strip()

            if spec:
                cleaned.append(spec)

        return cleaned

import re


def extract_tender_metadata(tender_text: str) -> dict:

    metadata = {
        "quantity": 1,
        "ministry": "Unknown",
        "department": "Unknown",
        "organisation": "Unknown",
        "office": "Unknown",
        "state": "Unknown",
        "product_category": "Unknown",
    }

    lines = [
        line.strip()
        for line in tender_text.splitlines()
        if line.strip()
    ]

    def value_after(label):

        for i, line in enumerate(lines):

            if label.lower() in line.lower():

                for j in range(i + 1, min(i + 4, len(lines))):

                    value = lines[j].strip()

                    if (
                        value
                        and "name" not in value.lower()
                        and label.lower() not in value.lower()
                    ):
                        return value

        return None

    metadata["ministry"] = (
        value_after("Ministry")
        or metadata["ministry"]
    )

    metadata["department"] = (
        value_after("Department")
        or metadata["department"]
    )

    metadata["organisation"] = (
        value_after("Organisation")
        or metadata["organisation"]
    )

    metadata["office"] = (
        value_after("Office")
        or metadata["office"]
    )

    metadata["state"] = (
        value_after("State")
        or metadata["state"]
    )

    quantity = re.search(
        r"Quantity\s*:?\s*(\d+)",
        tender_text,
        re.IGNORECASE,
    )

    if quantity:

        metadata["quantity"] = int(quantity.group(1))

    product = re.search(
        r"Item Category\s*:?\s*(.+)",
        tender_text,
        re.IGNORECASE,
    )

    if product:

        metadata["product_category"] = product.group(1).strip()

    return metadata

# ─────────────────────────────────────────────────────────────
# SECTION 2: VENDOR PROFILES
# These are programmatically generated — not from any external source.
# Realistic variances that test different compliance scenarios.
# ─────────────────────────────────────────────────────────────

VENDOR_PROFILES = [
    {
        "name": "Vendor A — TechSupply India Pvt Ltd",
        "turnover_lakhs": 100.0,
        "experience_years": 5,
        "is_mse": True,
        "is_mii": True,
        "emd_capacity_lakhs": 5.0,
        "delivery_days": 21,
        "supported_specs": [
            "16 GB RAM",
            "512 GB SSD",
            "Apple M4",
            "13.6 inch Display",
            "WiFi 6"
        ]
    },

    {
        "name": "Vendor B — GlobalIT Solutions",
        "turnover_lakhs": 30.0,
        "experience_years": 2,
        "is_mse": False,
        "is_mii": False,
        "emd_capacity_lakhs": 2.0,
        "delivery_days": 60,
        "supported_specs": [
            "16 GB RAM",
            "512 GB SSD",
            "Intel Core i7",
        ]
    },

    {
        "name": "Vendor C — BharatTech Systems",
        "turnover_lakhs": 250.0,
        "experience_years": 8,
        "is_mse": True,
        "is_mii": False,
        "emd_capacity_lakhs": 10.0,
        "delivery_days": 30,
        "supported_specs": [

            "16 GB RAM",
            "256 GB SSD",
            "Intel Core i7",
            "Windows 11 Pro",

]
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
- Extract ONLY measurable technical specifications.

GOOD examples:
- 16 GB RAM
- 512 GB SSD
- Apple M4
- Intel Core i7
- Windows 11 Pro
- Touchscreen

DO NOT extract:
- Product names
- Product titles
- Marketing descriptions
- Generic phrases like "Laptop"
- "High End Laptop"
- "Convertible Laptop"

Only return specifications that can be directly matched against a vendor's supported_specs.

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
            model="llama-3.3-70b-versatile",   # free tier, fast, sufficient for extraction
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


    # Check 5: Technical specifications
    vendor_specs = [
        spec.lower().strip()
        for spec in vendor["supported_specs"]
    ]

    missing_specs = []

    for required in requirements.product_specs:

        required = required.lower().strip()

        matched = False

        for vendor_spec in vendor_specs:

            if (
                required in vendor_spec
                or vendor_spec in required
            ):
                matched = True
                break

        if not matched:
            missing_specs.append(required)

    checks["technical"] = {
        "required": requirements.product_specs,
        "missing": missing_specs,
        "pass": len(missing_specs) == 0,
    }

    if missing_specs:
        failures.append(f"Missing technical specs: {', '.join(missing_specs)}")

    # Check 6: Delivery days
    delivery_pass = vendor["delivery_days"] <= requirements.delivery_days
    checks["delivery"] = {
        "required": f"{requirements.delivery_days} days",
        "vendor_has": f"{vendor['delivery_days']} days",
        "pass": delivery_pass
    }

    if not delivery_pass:
        failures.append(
            f"Delivery {vendor['delivery_days']} days exceeds limit {requirements.delivery_days} days"
        )

    # Update eligibility and summary
    is_eligible = len(failures) == 0



    return {
        "vendor_name": vendor["name"],
        "is_eligible": is_eligible,
        "checks": checks,
        "failures": failures,
        "mse_advantage": mse_advantage,
        "summary": "ELIGIBLE ✓" if is_eligible else f"INELIGIBLE ✗ — {'; '.join(failures)}"
    }


def run_compliance_check(tender_text: str):
    """
    Full pipeline:
    1. Extract compliance requirements (LLM)
    2. Extract tender metadata (deterministic parser)
    3. Evaluate vendors
    """

    requirements = extract_requirements_with_llm(tender_text)

    metadata = extract_tender_metadata(tender_text)

    logger.info(f"Extracted requirements: {requirements.model_dump()}")
    logger.info(f"Extracted metadata: {metadata}")

    results = []

    for vendor in VENDOR_PROFILES:

        result = evaluate_vendor(vendor, requirements)

        results.append(result)

        logger.info(f"{vendor['name']}: {result['summary']}")

    return {

        "requirements": requirements.model_dump(),

        "metadata": metadata,

        "vendors": results,

    }

if __name__ == "__main__":

    from data_pipeline import DataPipeline

    pipeline = DataPipeline().run()

    # Pick the first loaded tender
    tender_name = pipeline.get_tender_names()[0]

    tender_text = pipeline.get_tender_text(tender_name)

    print("=" * 80)
    print(f"Running Compliance Engine on: {tender_name}")
    print("=" * 80)

    result = run_compliance_check(tender_text)

    print("\n\nFINAL RESULT\n")
    print(result)