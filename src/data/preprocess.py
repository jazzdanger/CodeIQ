"""
Data Preprocessing & Code Filtering Module
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

CRITICAL SAFETY POLICY:
Source code submissions are strictly treated as untrusted text.
Code is NEVER executed. All validations are performed via static AST parsing.
"""

import ast
import hashlib
import re
from typing import Dict, Any, Tuple, Optional, Set
from src.config import settings
from src.utils.logger import logger


class CodePreprocessor:
    """
    Cleans, sanitizes, and filters raw source code submissions.
    Ensures complete, syntactically valid, non-duplicate Python submissions.
    """

    MIN_LINES = 2
    MAX_LINES = 500
    MAX_BYTES = settings.MAX_CODE_SIZE_BYTES  # 50,000 bytes

    def __init__(self):
        self._seen_hashes: Set[str] = set()

    @staticmethod
    def normalize_code_whitespace(source_code: str) -> str:
        """
        Normalizes line endings and trailing whitespace while preserving indentation.
        """
        lines = source_code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        cleaned_lines = [line.rstrip() for line in lines]
        # Remove leading/trailing blank lines
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        return "\n".join(cleaned_lines)

    @classmethod
    def compute_code_hash(cls, source_code: str) -> str:
        """
        Computes SHA-256 hash of normalized code for deduplication.
        Ignores pure comment and whitespace differences.
        """
        # Strip comments for duplicate detection
        stripped_lines = []
        for line in source_code.splitlines():
            line_content = line.split("#")[0].strip()
            if line_content:
                stripped_lines.append(line_content)
        canonical = "\n".join(stripped_lines)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def validate_code(cls, source_code: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates source code against size, line count, and static AST syntax boundaries.
        
        Returns:
            Tuple of (is_valid: bool, reason: str, metadata: dict)
        """
        metadata: Dict[str, Any] = {
            "num_lines": 0,
            "byte_size": 0,
            "syntax_valid": False,
        }

        if not isinstance(source_code, str):
            return False, "Input source code is not a string", metadata

        norm_code = cls.normalize_code_whitespace(source_code)
        byte_size = len(norm_code.encode("utf-8", errors="replace"))
        lines = norm_code.splitlines()
        num_lines = len(lines)

        metadata["num_lines"] = num_lines
        metadata["byte_size"] = byte_size

        # 1. Non-empty check
        if not norm_code.strip() or num_lines < cls.MIN_LINES:
            return False, f"Code too short (fewer than {cls.MIN_LINES} lines)", metadata

        # 2. Maximum limits check (Anti-DoS)
        if byte_size > cls.MAX_BYTES:
            return False, f"Code size ({byte_size} bytes) exceeds limit ({cls.MAX_BYTES} bytes)", metadata

        if num_lines > cls.MAX_LINES:
            return False, f"Line count ({num_lines}) exceeds limit ({cls.MAX_LINES})", metadata

        # 3. Static AST Syntax Parsing (Safe - zero execution)
        try:
            tree = ast.parse(norm_code)
            metadata["syntax_valid"] = True
            # Check if tree actually has meaningful statements
            if not tree.body:
                return False, "AST parsed to empty body", metadata
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}", metadata
        except Exception as e:
            return False, f"AST parse exception: {str(e)}", metadata

        return True, "Valid", metadata

    def filter_and_deduplicate(self, sample_id: str, source_code: str) -> Tuple[bool, str, str]:
        """
        Validates and checks submission against duplicate submissions in session.
        
        Returns:
            Tuple of (passed: bool, reason: str, normalized_code: str)
        """
        is_valid, reason, _ = self.validate_code(source_code)
        if not is_valid:
            return False, reason, ""

        norm_code = self.normalize_code_whitespace(source_code)
        code_hash = self.compute_code_hash(norm_code)

        if code_hash in self._seen_hashes:
            return False, "Duplicate submission detected", ""

        self._seen_hashes.add(code_hash)
        return True, "Accepted", norm_code

    def reset_deduplication(self) -> None:
        """Clears deduplication hash cache."""
        self._seen_hashes.clear()
