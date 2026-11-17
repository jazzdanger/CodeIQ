"""
Input Sanitization & Validation Module
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

CRITICAL SAFETY POLICY:
Source code submissions are strictly treated as untrusted text input.
Under NO circumstances is student code executed (no eval, no exec, no subprocess execution).
Analysis is purely static through syntax parsing and tokenization.
"""

import ast
from typing import Dict, Any, Tuple, Optional
from src.config import settings
from src.utils.logger import logger


class CodeSanitizer:
    """
    Validates and prepares untrusted source code for static AST parsing and embedding extraction.
    """

    ALLOWED_LANGUAGES = {"python", "py"}

    @classmethod
    def validate_submission(
        cls,
        source_code: str,
        language: str = "python"
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates untrusted source code against security constraints, size boundaries,
        and syntax validity.
        
        Args:
            source_code: Raw source code string.
            language: Programming language specified.
            
        Returns:
            Tuple of (is_valid: bool, error_message: str, metadata: dict)
        """
        metadata: Dict[str, Any] = {
            "byte_size": 0,
            "line_count": 0,
            "char_count": 0,
            "language": language.lower().strip(),
            "syntax_valid": False,
        }

        # 1. Null / Type check
        if not isinstance(source_code, str):
            return False, "Source code must be a string.", metadata

        # 2. Size limits check
        encoded_bytes = source_code.encode("utf-8", errors="replace")
        byte_size = len(encoded_bytes)
        metadata["byte_size"] = byte_size
        metadata["char_count"] = len(source_code)

        if byte_size == 0 or not source_code.strip():
            return False, "Source code submission cannot be empty.", metadata

        if byte_size > settings.MAX_CODE_SIZE_BYTES:
            msg = (
                f"Source code exceeds maximum permitted size of "
                f"{settings.MAX_CODE_SIZE_BYTES} bytes (Received: {byte_size} bytes)."
            )
            return False, msg, metadata

        # 3. Line count check
        lines = source_code.splitlines()
        metadata["line_count"] = len(lines)

        # 4. Language check
        lang_normalized = language.lower().strip()
        if lang_normalized not in cls.ALLOWED_LANGUAGES:
            msg = (
                f"Unsupported programming language '{language}'. "
                f"Supported languages: {list(cls.ALLOWED_LANGUAGES)}"
            )
            return False, msg, metadata

        # 5. Static Syntax Validation via ast.parse (Safe - does NOT execute code)
        try:
            ast.parse(source_code)
            metadata["syntax_valid"] = True
        except SyntaxError as e:
            metadata["syntax_valid"] = False
            msg = f"SyntaxError at line {e.lineno}, column {e.offset}: {e.msg}"
            return False, msg, metadata
        except Exception as e:
            metadata["syntax_valid"] = False
            msg = f"Failed to parse source code AST: {str(e)}"
            return False, msg, metadata

        return True, "", metadata
