from __future__ import annotations

import ast
from pathlib import Path


def test_final_parallel_cli_exposes_no_hyperparameter_override() -> None:
    source = Path("run_v2_final_parallel.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    option_strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        option_strings.append(arg.value)
    assert "--radius" not in option_strings
    assert "--alpha" not in option_strings
    assert "--workers" in option_strings
    assert "--max-runs" in option_strings
    assert "--workspace" in option_strings
