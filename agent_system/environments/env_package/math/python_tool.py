"""Run one Python calculation in a bounded, fresh subprocess."""

import ast
import os
import re
import signal
import subprocess
import sys
import tempfile


class PythonTool:
    def __init__(self, timeout=10, max_output_chars=4000, max_code_chars=12000, memory_mb=1024):
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.max_code_chars = max_code_chars
        self.memory_mb = memory_mb

    def execute(self, code):
        if not code.strip():
            return "Python error: empty code"
        if len(code) > self.max_code_chars:
            return "Python error: code is too long"
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Python syntax error: {exc.msg}"
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            expression = tree.body[-1].value
            if not (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id == "print"):
                tree.body[-1] = ast.Expr(value=ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[expression], keywords=[]))
                ast.fix_missing_locations(tree)
        limits = (
            "import resource\n"
            f"resource.setrlimit(resource.RLIMIT_CPU, ({self.timeout + 1}, {self.timeout + 2}))\n"
            f"resource.setrlimit(resource.RLIMIT_AS, ({self.memory_mb * 1024**2},) * 2)\n"
            "resource.setrlimit(resource.RLIMIT_FSIZE, (1048576,) * 2)\n"
        )
        preamble = "import math, cmath, statistics, itertools\nfrom fractions import Fraction\n"
        if re.search(r"\b(np|numpy)\b", code):
            preamble += "import numpy as np\nimport numpy\n"
        if re.search(r"\b(sp|sympy)\b", code):
            preamble += "import sympy as sp\nimport sympy\n"
        source = limits + preamble + ast.unparse(tree)

        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", source],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=errors,
                start_new_session=True,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            )
            try:
                process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return "Python error: execution timed out"
            output.seek(0)
            errors.seek(0)
            stdout = output.read(self.max_output_chars + 1).decode("utf-8", errors="replace")
            stderr = errors.read(self.max_output_chars + 1).decode("utf-8", errors="replace")
        result = stdout if process.returncode == 0 else f"Python error: {stderr.strip()}"
        return result.strip()[: self.max_output_chars] or "(no output)"
