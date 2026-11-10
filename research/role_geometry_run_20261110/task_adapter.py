"""Frozen-task adapters for the six-group role-geometry pilot.

Generation sees item["question"] only. HumanEval+ test bodies and reference
solutions live in the private dependency bundle and are scored after generation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import runpy
from types import SimpleNamespace

DATASET_TASKS = {
    "aime2024": "aime2024", "aime24": "aime2024",
    "gpqa": "gpqa", "gpqa_diamond": "gpqa", "gpqa_non_diamond": "gpqa",
    "humanevalplus": "humanevalplus", "humaneval_plus": "humanevalplus",
    "medqa": "medqa", "mmlu_pro": "gpqa",
}
CHOICE_DATASETS = frozenset(("gpqa", "gpqa_diamond", "gpqa_non_diamond", "medqa", "mmlu_pro"))
ROLES = ("planner", "critic", "refiner", "judger")


def task_for(item_or_dataset):
    dataset = item_or_dataset["dataset"] if isinstance(item_or_dataset, dict) else item_or_dataset
    if dataset not in DATASET_TASKS:
        raise ValueError(f"unsupported pilot dataset: {dataset}")
    task = DATASET_TASKS[dataset]
    if isinstance(item_or_dataset, dict) and item_or_dataset.get("task", task) != task:
        raise ValueError("dataset/task mismatch")
    return task


def args_for(dataset, model_name, method="latent_mas"):
    return SimpleNamespace(task=task_for(dataset), model_name=model_name,
        prompt="sequential", think=False, method=method,
        text_mas_context_length=-1, latent_space_realign=False)


def messages(role, item, build, model_name):
    if role not in ROLES:
        raise ValueError(f"unsupported role: {role}")
    task_for(item)
    args = args_for(item["dataset"], model_name)
    question = item["question"]
    if item["dataset"] != "mmlu_pro":
        return build(role=role, question=question, context="", method="latent_mas", args=args)
    # Patch only template boilerplate, never question text or option contents.
    sentinel = "__ROLE_GEOMETRY_EXACT_QUESTION_SLOT__"
    if sentinel in question:
        raise ValueError("question contains adapter sentinel")
    result = build(role=role, question=sentinel, context="", method="latent_mas", args=args)
    if role == "judger":
        n = int(item["n_options"])
        if not 2 <= n <= 26:
            raise ValueError("invalid option count")
        old = "Your final answer must be selected from A,B,C,D."
        new = "Your final answer must be selected from " + ",".join(chr(65+i) for i in range(n)) + "."
        if sum(message["content"].count(old) for message in result) != 1:
            raise ValueError("frozen MMLU template range was not found exactly once")
        result = [dict(message, content=message["content"].replace(old, new)) for message in result]
    return [dict(message, content=message["content"].replace(sentinel, question)) for message in result]


def last_box(text):
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return ""


def canonical_integer(value):
    if value is None:
        return None
    text = str(value).strip().replace("−", "-")
    if re.fullmatch(r"[-+]?\d+", text) is None:
        return None
    integer = int(text)
    return str(integer) if 0 <= integer <= 999 else None


def extract_python(text):
    # Same last-python-fence rule as pinned LatentMAS utils.py.
    blocks = re.findall(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return blocks[-1].strip() if blocks else None


def parse_answer(item, text, native_extract=None):
    task = task_for(item)
    if task == "humanevalplus":
        return extract_python(text)
    if task == "aime2024":
        boxed = last_box(text)
        if boxed is not None:
            return canonical_integer(boxed)
        if native_extract is not None:
            return canonical_integer(native_extract(text))
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        return canonical_integer(numbers[-1]) if numbers else None
    raw = native_extract(text) if native_extract is not None else last_box(text)
    if raw is None:
        return None
    pred = str(raw).strip().upper()
    allowed = [chr(65+i) for i in range(int(item["n_options"]))]
    return pred if pred in allowed else None


def normalized_gold(item):
    task = task_for(item)
    if task == "humanevalplus":
        return None
    if task == "aime2024":
        return canonical_integer(item["gold"])
    return str(item["gold"]).strip().upper()


def score_prediction(item, prediction, private_tests=None, executor=None, timeout=10):
    """Explicit post-generation scorer; this function never imports a GPU/model.

    HumanEval+ requires an isolated execution callback ``executor(code,test,timeout)``.
    The locked HF test body is the LatentMAS-paper evaluator, not the EvalPlus CLI.
    """
    if task_for(item) != "humanevalplus":
        return {"passed": prediction is not None and prediction == normalized_gold(item),
                "scorer": "aime_integer_exact" if task_for(item) == "aime2024" else "choice_exact"}
    if prediction is None:
        return {"passed": False, "error_type": "InvalidGeneration", "scorer": "LatentMAS_HF_HumanEvalPlus_tests"}
    if private_tests is None or executor is None:
        raise ValueError("HumanEval+ requires locked private tests and an isolated executor")
    reference = private_tests[item["private_test_key"]]
    if reference["entry_point"] != item["entry_point"]:
        raise ValueError("entry point mismatch")
    test = reference["rendered_test"]
    if hashlib.sha256(test.encode("utf-8")).hexdigest() != reference["rendered_test_sha256"]:
        raise ValueError("HumanEval+ test hash mismatch")
    result = dict(executor(prediction, test, timeout))
    result["scorer"] = "LatentMAS_HF_HumanEvalPlus_tests"
    return result


def load_frozen_builder(data_root):
    root = Path(data_root)
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    relative = "dependencies/LatentMAS/prompts.py"
    path = root / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["dependencies"][relative]["sha256"]:
        raise ValueError("frozen prompt source SHA mismatch")
    return runpy.run_path(str(path))["build_agent_message_sequential_latent_mas"]
