"""
Structured Response Schema Validator & Repair
VeriProof AI Infrastructure Foundation
"""
import json
import re
import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger("ai_infrastructure.schema_validator")

class SchemaValidationError(Exception):
    pass

class ResponseValidator:
    @classmethod
    def clean_json_text(cls, text: str) -> str:
        text = text.strip()
        # Remove markdown fence blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @classmethod
    def validate_and_parse_json(cls, raw_response: str) -> Tuple[bool, Any, str]:
        cleaned = cls.clean_json_text(raw_response)
        try:
            parsed = json.loads(cleaned)
            return True, parsed, ""
        except json.JSONDecodeError as err:
            logger.warning(f"JSON Parse Error: {err.msg}. Attempting regex JSON repair...")
            # Try finding first { or [ and last } or ]
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if match:
                try:
                    repaired = json.loads(match.group(1))
                    logger.info("JSON Regex repair succeeded!")
                    return True, repaired, ""
                except json.JSONDecodeError:
                    pass
            return False, None, f"JSON decode failed: {err.msg}"
