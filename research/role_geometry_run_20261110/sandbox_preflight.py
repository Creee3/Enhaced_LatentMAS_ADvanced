"""CPU-only isolated HumanEval+ evaluator preflight; never imports a model."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from common import HERE, load, save, sha
from task_adapter import parse_answer, score_prediction
from evaluator_overlay import TASK as OVERLAY_TASK, make_executor, verify_lock as verify_overlay, witness_code


def load_executor(root):
    root = Path(root)
    manifest = load(root/"data_manifest.json")
    required = ("dependencies/scoring/sandbox.py", "dependencies/scoring/sandbox_child.py", "dependencies/humanevalplus_tests.json")
    for relative in required:
        expected = manifest["dependencies"][relative]
        path = root/relative
        if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("private executor dependency SHA mismatch")
    relative = "executor.py"
    spec = importlib.util.spec_from_file_location("role_geometry_frozen_sandbox", root/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return make_executor(module.run,root)


def run_preflight(root, executor):
    root = Path(root)
    manifest = load(root/"data_manifest.json")
    if sha(root/"questions.json") != manifest["questions_sha256"]:
        raise ValueError("question SHA mismatch")
    questions = [row for row in load(root/"questions.json") if row["dataset"] == "humanevalplus"]
    tests = load(root/"dependencies/humanevalplus_tests.json")
    if len(questions) != 20 or len(tests) != 20:
        raise ValueError("HumanEval+ preflight requires all selected20")
    timeout = manifest["humaneval_scorer"]["timeout_seconds"]
    if timeout != 10:
        raise ValueError("paper evaluator timeout changed")
    overlay_lock = verify_overlay(root)
    started = time.time()
    details = []
    for item in questions:
        reference = tests[item["private_test_key"]]
        canonical = reference["prompt"] + reference["canonical_solution"]
        ast.parse(canonical)
        wrong = f"def {item['entry_point']}(*args, **kwargs):\n    raise AssertionError('intentional evaluator negative fixture')\n"
        begin = time.monotonic()
        positive = score_prediction(item, canonical, tests, executor, timeout)
        negative = score_prediction(item, wrong, tests, executor, timeout)
        declared_numeric_failure = item["question_id"] == OVERLAY_TASK and not positive.get("passed") and positive.get("error_type") == "AssertionError"
        witness = score_prediction(item,witness_code(root),tests,executor,timeout) if declared_numeric_failure else None
        positive_gate = bool(positive.get("passed")) or (declared_numeric_failure and bool(witness.get("passed")))
        outcomes = (positive,negative) if witness is None else (positive,negative,witness)
        infrastructure = [result.get("error_type") for result in outcomes if result.get("error_type") == "SandboxFailure"]
        detail = dict(task_id=item["question_id"], canonical_passed=bool(positive.get("passed")),
            wrong_script_rejected=not bool(negative.get("passed")),
            positive_gate_passed=positive_gate, declared_canonical_numeric_failure=declared_numeric_failure,
            independent_witness_passed=bool(witness and witness.get("passed")),
            evaluator_overlay=positive.get("evaluator_overlay"),
            positive_error_type=positive.get("error_type"), negative_error_type=negative.get("error_type"),
            infrastructure_error=bool(infrastructure), seconds=time.monotonic()-begin,
            canonical_code_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
            rendered_test_sha256=reference["rendered_test_sha256"])
        details.append(detail)
    parser_negative = all(parse_answer(item, "No fenced Python code in this negative fixture.") is None for item in questions)
    canonical_passed = sum(detail["canonical_passed"] for detail in details)
    positive_gates = sum(detail["positive_gate_passed"] for detail in details)
    known_failures = [detail["task_id"] for detail in details if detail["declared_canonical_numeric_failure"]]
    witnesses = sum(detail["independent_witness_passed"] for detail in details)
    script_negatives = sum(detail["wrong_script_rejected"] for detail in details)
    infrastructure_failures = sum(detail["infrastructure_error"] for detail in details)
    return dict(passed=positive_gates == 20 and script_negatives == 20 and parser_negative and not infrastructure_failures and known_failures in ([],[OVERLAY_TASK]),
        model_forward_calls=0, questions=20, canonical_passed=canonical_passed,
        positive_gate_passed=positive_gates, known_canonical_numeric_failures=known_failures,
        independent_witness_passed=witnesses,
        evaluator_overlay_lock_sha256=sha(root/"evaluator_overlay_lock.json"),
        evaluator_overlay_source_sha256=sha(root/"evaluator_overlay.py"),
        witness_sha256=overlay_lock["witness"]["sha256"],
        wrong_script_rejected=script_negatives, parser_negative_passed=parser_negative,
        infrastructure_failures=infrastructure_failures, isolated_execution=True,
        evaluator="LatentMAS_HF_tests_with_declared_HE32_official_predicate_order_overlay", evalplus_cli_used=False,
        timeout_seconds=timeout, started_unix=started, finished_unix=time.time(),
        questions_sha256=manifest["questions_sha256"], tests_sha256=sha(root/"dependencies/humanevalplus_tests.json"),
        source_sha256=sha(Path(__file__)), dependencies={relative:manifest["dependencies"][relative] for relative in
            ("dependencies/scoring/sandbox.py", "dependencies/scoring/sandbox_child.py", "dependencies/humanevalplus_tests.json")},
        details=details)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=HERE)
    args = parser.parse_args()
    root = args.run_root.resolve()
    try:
        executor = load_executor(root)
        result = run_preflight(root, executor)
    except Exception as error:
        # Private evidence has only an exception type; no test/code/log disclosure.
        result = dict(passed=False, model_forward_calls=0, infrastructure_error_type=type(error).__name__,
                      source_sha256=sha(Path(__file__)), unix=time.time())
    save(root/"executor_preflight.json", result)
    print(json.dumps({key:result[key] for key in ("passed","model_forward_calls","questions","canonical_passed","positive_gate_passed","independent_witness_passed","wrong_script_rejected","parser_negative_passed","infrastructure_failures") if key in result}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
