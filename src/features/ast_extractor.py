"""
AST Feature Extraction Module
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Extracts 25 structural, control-flow, and Halstead complexity features from Python source code
using static syntax trees (ast) and Radon metrics.

SAFETY & DESIGN GUARANTEES:
- Strictly static analysis: ZERO code execution (no eval, exec, or subprocess).
- Bounded recursion depth to prevent stack overflows on malicious deeply nested inputs.
- Handles syntax errors, empty files, and malformed inputs gracefully with default vectors.
- Measures and records extraction latency (in milliseconds).
"""

import ast
import time
import math
from typing import Dict, Any, List, Tuple, Set, Optional
import numpy as np
from radon.complexity import cc_visit
import radon.metrics as rm

from src.config import settings
from src.utils.logger import logger


class ASTFeatureExtractor:
    """
    Analyzes Python Abstract Syntax Trees to extract 25 structural and algorithmic metrics.
    """

    FEATURE_NAMES: List[str] = settings.AST_FEATURE_NAMES

    def __init__(self, max_recursion_depth: int = 250):
        self.max_recursion_depth = max_recursion_depth

    def extract_features(self, source_code: str) -> Dict[str, Any]:
        """
        Extracts all 25 features from source code and measures execution latency.
        
        Args:
            source_code: Python source code string.
            
        Returns:
            Dictionary containing:
            - 'features': Dict[str, float] of 25 feature values
            - 'vector': np.ndarray of shape (25,)
            - 'extraction_time_ms': float latency in milliseconds
            - 'success': bool
            - 'error': Optional[str]
        """
        start_time = time.perf_counter()
        
        # 1. Handle empty input
        if not source_code or not source_code.strip():
            latency = (time.perf_counter() - start_time) * 1000.0
            return self._empty_result(latency=latency, error="Empty source code")

        # 2. Parse AST safely
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            return self._empty_result(latency=latency, error=f"SyntaxError: {e.msg} at line {e.lineno}")
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            return self._empty_result(latency=latency, error=f"AST parse failure: {str(e)}")

        # 3. Extract metrics via AST traversal
        try:
            features = self._compute_ast_metrics(tree, source_code)
            vector = np.array([features[name] for name in self.FEATURE_NAMES], dtype=np.float32)
            latency = (time.perf_counter() - start_time) * 1000.0

            return {
                "features": features,
                "vector": vector,
                "extraction_time_ms": round(latency, 3),
                "success": True,
                "error": None
            }
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            logger.warning(f"Error computing AST metrics: {e}")
            return self._empty_result(latency=latency, error=str(e))

    def _compute_ast_metrics(self, tree: ast.AST, source_code: str) -> Dict[str, float]:
        """Traverses AST to compute the exact 25 structural and complexity metrics."""
        
        # 1. Total AST nodes
        all_nodes = list(ast.walk(tree))
        total_ast_nodes = float(len(all_nodes))

        # 2. Number of functions
        functions = [n for n in all_nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        num_functions = float(len(functions))

        # 3. Number of classes
        classes = [n for n in all_nodes if isinstance(n, ast.ClassDef)]
        num_classes = float(len(classes))

        # 4. Number of variables (Unique stored identifiers)
        assigned_vars = set()
        all_var_names = []
        for n in all_nodes:
            if isinstance(n, ast.Name):
                all_var_names.append(n.id)
                if isinstance(n.ctx, ast.Store):
                    assigned_vars.add(n.id)
        num_variables = float(len(assigned_vars))

        # 5. Number of loops
        loop_types = (ast.For, ast.AsyncFor, ast.While)
        loops = [n for n in all_nodes if isinstance(n, loop_types)]
        num_loops = float(len(loops))

        # 6 & 7. Loop nesting depth & nested loop counts
        num_nested_loops, max_loop_nesting_depth = self._analyze_loop_nesting(tree)

        # 8. Cyclomatic complexity (McCabe)
        cyclomatic_complexity = self._compute_cyclomatic_complexity(source_code, tree)

        # 9. Number of conditional statements (if / elif)
        conditionals = [n for n in all_nodes if isinstance(n, ast.If)]
        num_conditional_statements = float(len(conditionals))

        # 10. Number of branching conditions (and, or, if-expressions)
        bool_ops = sum(len(n.values) - 1 for n in all_nodes if isinstance(n, ast.BoolOp))
        if_exps = sum(1 for n in all_nodes if isinstance(n, ast.IfExp))
        num_branching_conditions = float(bool_ops + if_exps)

        # 11. Number of return statements
        returns = [n for n in all_nodes if isinstance(n, ast.Return)]
        num_return_statements = float(len(returns))

        # 12. Number of recursive functions
        num_recursive_functions = float(self._count_recursive_functions(functions))

        # 13. Number of function calls
        calls = [n for n in all_nodes if isinstance(n, ast.Call)]
        num_function_calls = float(len(calls))

        # 14. Number of parameters across all functions
        num_parameters = float(sum(
            len(fn.args.args) + (1 if fn.args.vararg else 0) + (1 if fn.args.kwarg else 0)
            for fn in functions
        ))

        # 15 & 16. Function lengths (statements)
        if functions:
            fn_lengths = [len(fn.body) for fn in functions]
            avg_function_length = float(np.mean(fn_lengths))
            max_function_length = float(np.max(fn_lengths))
        else:
            avg_function_length = 0.0
            max_function_length = 0.0

        # 17 - 23. Operator, Operand, and Halstead Metrics
        halstead = self._compute_halstead_metrics(all_nodes, source_code)

        # 24. Variable reuse ratio: total variable references / unique variables
        unique_vars_count = max(1, len(set(all_var_names)))
        variable_reuse_ratio = float(len(all_var_names) / unique_vars_count)

        # 25. Control-flow branching factor: (Branches + Loops) / Max(1, Functions)
        total_decision_points = num_loops + num_conditional_statements + num_branching_conditions
        control_flow_branching_factor = float(total_decision_points / max(1.0, num_functions))

        return {
            "total_ast_nodes": total_ast_nodes,
            "num_functions": num_functions,
            "num_classes": num_classes,
            "num_variables": num_variables,
            "num_loops": num_loops,
            "num_nested_loops": float(num_nested_loops),
            "max_loop_nesting_depth": float(max_loop_nesting_depth),
            "cyclomatic_complexity": cyclomatic_complexity,
            "num_conditional_statements": num_conditional_statements,
            "num_branching_conditions": num_branching_conditions,
            "num_return_statements": num_return_statements,
            "num_recursive_functions": num_recursive_functions,
            "num_function_calls": num_function_calls,
            "num_parameters": num_parameters,
            "avg_function_length": avg_function_length,
            "max_function_length": max_function_length,
            "num_operators": float(halstead["num_operators"]),
            "num_operands": float(halstead["num_operands"]),
            "halstead_vocabulary": float(halstead["vocabulary"]),
            "halstead_length": float(halstead["length"]),
            "halstead_volume": float(halstead["volume"]),
            "halstead_difficulty": float(halstead["difficulty"]),
            "halstead_effort": float(halstead["effort"]),
            "variable_reuse_ratio": round(variable_reuse_ratio, 3),
            "control_flow_branching_factor": round(control_flow_branching_factor, 3),
        }

    def _analyze_loop_nesting(self, root: ast.AST) -> Tuple[int, int]:
        """
        Recursively traverses AST hierarchy to calculate:
        - Total count of nested loops (loops inside loops)
        - Maximum loop nesting depth
        """
        nested_loops = 0
        max_depth = 0
        loop_types = (ast.For, ast.AsyncFor, ast.While)

        def traverse(node: ast.AST, current_depth: int):
            nonlocal nested_loops, max_depth
            if current_depth > self.max_recursion_depth:
                return

            is_loop = isinstance(node, loop_types)
            next_depth = current_depth + 1 if is_loop else current_depth

            if is_loop:
                if current_depth >= 1:
                    nested_loops += 1
                if next_depth > max_depth:
                    max_depth = next_depth

            for child in ast.iter_child_nodes(node):
                traverse(child, next_depth)

        traverse(root, 0)
        return nested_loops, max_depth

    @staticmethod
    def _count_recursive_functions(functions: List[ast.AST]) -> int:
        """Counts functions that directly call themselves."""
        recursive_count = 0
        for fn in functions:
            fn_name = getattr(fn, "name", None)
            if not fn_name:
                continue
            is_rec = False
            for child in ast.walk(fn):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == fn_name:
                        is_rec = True
                        break
            if is_rec:
                recursive_count += 1
        return recursive_count

    @staticmethod
    def _compute_cyclomatic_complexity(source_code: str, tree: ast.AST) -> float:
        """
        Computes McCabe Cyclomatic Complexity using Radon with AST decision fallback.
        """
        try:
            blocks = cc_visit(source_code)
            if blocks:
                return float(sum(b.complexity for b in blocks))
        except Exception:
            pass

        # Fallback: Decision point count + 1
        decisions = 1
        for n in ast.walk(tree):
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                decisions += 1
            elif isinstance(n, ast.BoolOp):
                decisions += len(n.values) - 1
            elif isinstance(n, ast.IfExp):
                decisions += 1
        return float(decisions)

    @staticmethod
    def _compute_halstead_metrics(all_nodes: List[ast.AST], source_code: str) -> Dict[str, float]:
        """
        Computes Halstead Software Science metrics:
        - n1 = unique operators
        - n2 = unique operands
        - N1 = total operators
        - N2 = total operands
        - Vocabulary = n1 + n2
        - Length = N1 + N2
        - Volume = Length * log2(Vocabulary)
        - Difficulty = (n1 / 2) * (N2 / n2)
        - Effort = Difficulty * Volume
        """
        operators: List[str] = []
        operands: List[str] = []

        for node in all_nodes:
            # Operators
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                operators.append(node.__class__.__name__)
            elif isinstance(node, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                operators.append(node.__class__.__name__)
            elif isinstance(node, (ast.And, ast.Or, ast.Not)):
                operators.append(node.__class__.__name__)
            elif isinstance(node, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert, ast.LShift, ast.RShift)):
                operators.append(node.__class__.__name__)
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Return, ast.Yield)):
                operators.append(node.__class__.__name__)

            # Operands
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                operands.append(f"var_{node.id}")
            elif isinstance(node, ast.Constant):
                operands.append(f"const_{type(node.value).__name__}_{node.value}")

        N1 = len(operators)
        N2 = len(operands)
        n1 = len(set(operators))
        n2 = len(set(operands))

        vocab = max(1, n1 + n2)
        length = N1 + N2
        volume = float(length * math.log2(max(2, vocab)))
        difficulty = float((n1 / 2.0) * (N2 / max(1.0, n2)))
        effort = float(difficulty * volume)

        return {
            "num_operators": float(N1),
            "num_operands": float(N2),
            "vocabulary": float(vocab),
            "length": float(length),
            "volume": round(volume, 2),
            "difficulty": round(difficulty, 2),
            "effort": round(effort, 2)
        }

    def _empty_result(self, latency: float, error: str) -> Dict[str, Any]:
        """Returns a consistent zeroed vector for failed/empty parses."""
        features = {name: 0.0 for name in self.FEATURE_NAMES}
        return {
            "features": features,
            "vector": np.zeros(len(self.FEATURE_NAMES), dtype=np.float32),
            "extraction_time_ms": round(latency, 3),
            "success": False,
            "error": error
        }
