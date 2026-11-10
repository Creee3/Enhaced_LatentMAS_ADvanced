import os
"""CPU-only adapter, private-data boundary and source-lock checks."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from task_adapter import (args_for, canonical_integer, extract_python, load_frozen_builder,
    messages, normalized_gold, parse_answer, score_prediction, task_for)

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ["ROLE_GEOMETRY_DATA_ROOT"]) if "ROLE_GEOMETRY_DATA_ROOT" in os.environ else HERE.parents[2] / "advanced_MOC/experiments/latentmas_role_geometry120_20261110"
BUILD = load_frozen_builder(DATA)
MODEL = "Qwen/Qwen3-8B"


def native(text):
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


class TaskAdapters(unittest.TestCase):
    def test_task_mapping_is_explicit(self):
        expected = {"aime2024":"aime2024", "gpqa":"gpqa", "humanevalplus":"humanevalplus", "medqa":"medqa", "mmlu_pro":"gpqa"}
        for dataset, task in expected.items():
            self.assertEqual(args_for(dataset, MODEL).task, task)
        with self.assertRaises(ValueError):
            task_for({"dataset":"aime2024", "task":"gpqa"})
        with self.assertRaises(ValueError):
            task_for("unsupported")

    def test_author_templates_remain_byte_exact(self):
        for dataset in ("aime2024", "gpqa", "humanevalplus", "medqa"):
            item = dict(dataset=dataset, question="Synthetic public problem only.", n_options=4)
            for role in ("planner", "critic", "refiner", "judger"):
                original = BUILD(role=role, question=item["question"], context="", method="latent_mas", args=args_for(dataset, MODEL))
                self.assertEqual(messages(role, item, BUILD, MODEL), original)

    def test_mmlu_option_patch_does_not_modify_question(self):
        item = dict(dataset="mmlu_pro", question="Literal Your final answer must be selected from A,B,C,D. within synthetic option.", n_options=10)
        for role in ("planner", "critic", "refiner", "judger"):
            result = messages(role, item, BUILD, MODEL)
            self.assertIn(item["question"], result[1]["content"])
            if role == "judger":
                self.assertEqual(result[1]["content"].count("selected from A,B,C,D,E,F,G,H,I,J."), 1)
        self.assertIsNone(parse_answer(item, r"\boxed{K}", native))

    def test_aime_is_integer_exact(self):
        item = dict(dataset="aime2024", gold="007")
        fixtures = {r"work 88 then \boxed{007}":"7", r"\boxed{999}":"999", r"\boxed{0}":"0",
            r"\boxed{3.0}":None, r"\boxed{-1}":None, r"\boxed{1000}":None,
            r"\boxed{1 or 2}":None, r"\boxed{\frac{3}{1}}":None}
        for text, want in fixtures.items():
            self.assertEqual(parse_answer(item, text, native), want)
        self.assertEqual(normalized_gold(item), "7")
        self.assertTrue(score_prediction(item, "7")["passed"])
        self.assertFalse(score_prediction(item, None)["passed"])

    def test_choice_ranges_and_case(self):
        for dataset, count in (("gpqa",4),("medqa",4),("mmlu_pro",10)):
            item = dict(dataset=dataset, gold="b", n_options=count)
            self.assertEqual(normalized_gold(item), "B")
            for index in range(count):
                value = chr(65+index)
                self.assertEqual(parse_answer(item, r"\boxed{"+value+"}", native), value)
            self.assertIsNone(parse_answer(item, r"\boxed{Z}", native))
            self.assertIsNone(parse_answer(item, r"\boxed{AB}", native))

    def test_humaneval_last_python_fence_only(self):
        item = dict(dataset="humanevalplus")
        text = "```python\ndef first(): return 0\n``` text ```Python\ndef final(): return 1\n```"
        self.assertEqual(parse_answer(item, text), "def final(): return 1")
        self.assertIsNone(parse_answer(item, "def unfenced(): return 1"))
        self.assertIsNone(normalized_gold(item))

    def test_humaneval_needs_private_executor_and_test_hash(self):
        test = "def check(f): assert f(1)==2\ncheck(f)"
        record = dict(entry_point="f", rendered_test=test, rendered_test_sha256=hashlib.sha256(test.encode()).hexdigest())
        tests = {"toy":record}
        item = dict(dataset="humanevalplus", entry_point="f", private_test_key="toy")
        captured = []
        def fake_executor(code, received_test, timeout):
            captured.append((code, received_test, timeout))
            return {"passed":True}
        self.assertTrue(score_prediction(item, "def f(x): return x+1", tests, fake_executor)["passed"])
        self.assertEqual(captured[0][1], test)
        with self.assertRaises(ValueError):
            score_prediction(item, "def f(x): return x+1", tests)
        tests["toy"]["rendered_test"] += "#tampered"
        with self.assertRaises(ValueError):
            score_prediction(item, "def f(x): return x+1", tests, fake_executor)

    def test_private_dataset_schema_and_counts(self):
        rows = json.loads((DATA/"questions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({row["question_id"] for row in rows}), 120)
        self.assertEqual(set(Counter(row["dataset_group"] for row in rows).values()), {20})
        required = {"index","global_index","dataset","dataset_group","question_id","question","task","n_options","gold","is_diamond","category"}
        for index, row in enumerate(rows):
            self.assertTrue(required.issubset(row))
            self.assertEqual(row["index"], index)
            self.assertEqual(row["global_index"], index)
            self.assertEqual(row["question_sha256"], hashlib.sha256(row["question"].encode()).hexdigest())
            self.assertFalse({"test","canonical_solution","gen_text_store","cot_content"}.intersection(row))
            for role in ("planner","critic","refiner","judger"):
                result = messages(role, row, BUILD, MODEL)
                self.assertEqual(sum(message["content"].count(row["question"]) for message in result), 1)

    def test_input_source_hashes_and_private_tests(self):
        manifest = json.loads((DATA/"data_manifest.json").read_text(encoding="utf-8"))
        for relative, expected in manifest["dependencies"].items():
            self.assertEqual(hashlib.sha256((DATA/relative).read_bytes()).hexdigest(), expected["sha256"])
        self.assertEqual(manifest["target_keys"], 1440)
        self.assertEqual(manifest["model_forward_calls"], 0)
        self.assertEqual(manifest["engineering_indices"], [0,20,40,60,80,100])
        tests = json.loads((DATA/"dependencies/humanevalplus_tests.json").read_text(encoding="utf-8"))
        self.assertEqual(len(tests), 20)
        rows = json.loads((DATA/"questions.json").read_text(encoding="utf-8"))
        for row in (row for row in rows if row["dataset"]=="humanevalplus"):
            private = tests[row["private_test_key"]]
            self.assertIsNone(row["gold"])
            self.assertNotIn(private["raw_test"], row["question"])
            self.assertNotIn(private["rendered_test"], row["question"])
            self.assertTrue(private["rendered_test"].endswith("check("+row["entry_point"]+")"))
            ast.parse(private["rendered_test"])

    def test_lock_covers_adapter_and_private_inputs(self):
        lock = json.loads((HERE/"adapter_source_lock.json").read_text(encoding="utf-8"))
        for name, expected in lock["source_files"].items():
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(), expected)
        for name, expected in lock["private_artifacts"].items():
            self.assertEqual(hashlib.sha256((DATA/name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
