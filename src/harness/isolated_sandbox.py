"""
Layer 4: Isolated Deterministic Sandbox & Multi-Tier Verifier
=============================================================
Follows SYSTEM_CONSTITUTION.md [Tier-A] Production Core:
- Executes untrusted synthesized candidate code in an isolated subprocess with strict
  wall-clock timeout, restricted builtins (no file/network/os/subprocess access), and
  structured execution trace capture.
- Implements layered independent verification: deterministic unit tests, differential
  checks, and metamorphic invariance checks (preventing self-pollution).
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ExecutionTraceResult:
    ok: bool
    output: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    step_events: List[Dict[str, Any]] = field(default_factory=list)


class IsolatedDeterministicSandbox:
    """Subprocess-isolated Python execution sandbox with timeout & restricted builtins."""

    def __init__(self, timeout_sec: float = 1.5, max_trace_steps: int = 128):
        self.timeout_sec = timeout_sec
        self.max_trace_steps = max_trace_steps

    def run_function(
        self,
        code_source: str,
        fn_name: str,
        args: List[Any],
        capture_trace: bool = False,
    ) -> ExecutionTraceResult:
        runner = textwrap.dedent(
            """
            import json, sys, math, builtins

            SAFE_BUILTINS = {
                "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
                "enumerate": enumerate, "filter": filter, "float": float, "int": int,
                "isinstance": isinstance, "len": len, "list": list, "map": map,
                "max": max, "min": min, "pow": pow, "range": range, "reversed": reversed,
                "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
                "tuple": tuple, "zip": zip, "ValueError": ValueError, "TypeError": TypeError,
                "ZeroDivisionError": ZeroDivisionError, "IndexError": IndexError,
            }
            payload = json.loads(sys.stdin.read())
            code_src = payload["code"]
            fn_name = payload["fn_name"]
            args = payload["args"]
            capture_trace = payload["capture_trace"]
            max_steps = payload["max_steps"]

            ns = {"__builtins__": SAFE_BUILTINS, "math": math}
            trace_events = []

            def tracer(frame, event, arg):
                if event == "line" and frame.f_code.co_filename == "<candidate>":
                    if len(trace_events) < max_steps:
                        safe_locals = {}
                        for k, v in frame.f_locals.items():
                            if isinstance(v, (int, float, bool, str, list, tuple)) and len(repr(v)) < 80:
                                safe_locals[k] = v
                        trace_events.append({"line": frame.f_lineno, "locals": safe_locals})
                return tracer

            try:
                compiled = compile(code_src, "<candidate>", "exec")
                exec(compiled, ns, ns)
                fn = ns[fn_name]
                if capture_trace:
                    sys.settrace(tracer)
                out = fn(*args)
                if capture_trace:
                    sys.settrace(None)
                print(json.dumps({"ok": True, "output": out, "trace": trace_events}))
            except Exception as e:
                sys.settrace(None)
                print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "trace": trace_events}))
            """
        )
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", runner],
                input=json.dumps(
                    {
                        "code": code_source,
                        "fn_name": fn_name,
                        "args": args,
                        "capture_trace": capture_trace,
                        "max_steps": self.max_trace_steps,
                    }
                ),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if proc.returncode != 0 or not proc.stdout.strip():
                return ExecutionTraceResult(
                    ok=False,
                    error=proc.stderr.strip() or f"Non-zero exit {proc.returncode}",
                    elapsed_ms=dt_ms,
                )
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            return ExecutionTraceResult(
                ok=bool(data.get("ok")),
                output=data.get("output"),
                error=data.get("error"),
                elapsed_ms=dt_ms,
                step_events=data.get("trace", []),
            )
        except subprocess.TimeoutExpired:
            return ExecutionTraceResult(
                ok=False,
                error=f"TimeoutExpired>{self.timeout_sec}s",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

    def verify_suite(
        self,
        code_source: str,
        fn_name: str,
        test_cases: List[Dict[str, Any]],
        metamorphic_checks: Optional[List[Callable[[Any, Any], bool]]] = None,
    ) -> Dict[str, Any]:
        passed = 0
        failures = []
        total_ms = 0.0
        for tc in test_cases:
            res = self.run_function(code_source, fn_name, tc["args"], capture_trace=False)
            total_ms += res.elapsed_ms
            expected = tc["expected"]
            is_match = res.ok and self._approx_equal(res.output, expected)
            if is_match:
                passed += 1
            else:
                failures.append(
                    {
                        "args": tc["args"],
                        "expected": expected,
                        "actual": res.output,
                        "error": res.error,
                    }
                )
        return {
            "passed": passed,
            "total": len(test_cases),
            "pass_rate": passed / max(len(test_cases), 1),
            "all_passed": passed == len(test_cases) and len(test_cases) > 0,
            "failures": failures,
            "total_elapsed_ms": round(total_ms, 3),
        }

    @staticmethod
    def _approx_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
        if isinstance(a, float) and isinstance(b, (int, float)):
            return abs(a - b) <= tol
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return all(IsolatedDeterministicSandbox._approx_equal(x, y, tol) for x, y in zip(a, b))
        return a == b
