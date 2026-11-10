"""Prepare the private six-group pilot inputs without tokenizer/model forwards."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DEFAULT_OUT = WORKSPACE / "advanced_MOC/experiments/latentmas_role_geometry120_20261110"
SELECTION = HERE.parent / "role_geometry_design_20261110/pilot_selection_manifest.json"
SELECTION_SHA = "7756c583be41780cd60ae9d88b04a562d99246069d85e654e22982ccb1f344e5"
SALT = "role_geometry_pilot_20261110_v1"
GROUPS = ("AIME24", "GPQA-Diamond", "GPQA-nonDiamond", "HumanEval+", "MedQA", "MMLU-Pro")
METHODS = ("Full", "Gen", "H", "L", "H-OBF", "L-OBF", "G", "T", "R", "R-shuffle", "Pseudo-R", "Random-matched")
AUTHOR = WORKSPACE / "04_参考代码与历史实验/上游参考代码/LatentMAS-official"
AUTHOR_COMMIT = "9a9e4d331eb11430bd9e64754c6b252b06d73031"
SANDBOX = WORKSPACE / "advanced_MOC/experiments/moc_v21_humaneval_test_20261015"
EXPECTED_AUTHOR = {
    "data.py": "ca00592eadd7b8c268f51c70bbfb49d8088fe387ea9952d288b65a6f5d80839a",
    "prompts.py": "8cf16184937f2d0bb78c856426685ccc8d5cd639e8df0f5c7642dfa87064e800",
    "utils.py": "f58309e2b549ac9c560aea3642e82f9406f3591d1c7c7c18c3f584faadb9bb69",
    "methods/latent_mas.py": "3fb7b5a7032c2ebe912a39b681ca6a19aec6664eb1742c71dfbca4cf5e164d33",
}
EXPECTED_SANDBOX = {
    "sandbox.py": "608619a4734f7765e15911176a884b080873cc3c66e66ec8a4a3cda79376e313",
    "sandbox_child.py": "e5943eab6bddf08e7f24117f718aac28fa8b41630e343666229acbf95662411f",
}
SOURCE_TARGETS = {
    "AIME24": "dependencies/source_inputs/aime24.parquet",
    "GPQA-Diamond": "dependencies/source_inputs/gpqa_cohort448.json",
    "GPQA-nonDiamond": "dependencies/source_inputs/gpqa_cohort448.json",
    "HumanEval+": "dependencies/source_inputs/humanevalplus.parquet",
    "MedQA": "dependencies/source_inputs/medqa.json",
    "MMLU-Pro": "dependencies/source_inputs/mmlu_pro.parquet",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() == content:
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def copy_exact(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    digest = sha(source)
    if expected is not None and digest != expected:
        raise ValueError(f"source SHA mismatch: {source.name}")
    if destination.exists():
        if sha(destination) != digest:
            raise ValueError(f"refusing to overwrite different frozen dependency: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if sha(destination) != digest:
        raise ValueError("dependency copy readback mismatch")
    return {"sha256": digest, "bytes": destination.stat().st_size}


def load_rows(path):
    path = Path(path)
    return pq.read_table(path).to_pylist() if path.suffix == ".parquet" else json.loads(path.read_text(encoding="utf-8"))


def rank(group, stable_id):
    return hashlib.sha256(f"{SALT}|{group}|{stable_id}".encode("utf-8")).hexdigest()


def verify_selection(selection, raw):
    if tuple(selection["selected"]) != GROUPS or selection["salt"] != SALT:
        raise ValueError("selection scope changed")
    all_ids = []
    for group in GROUPS:
        chosen = selection["selected"][group]
        if len(chosen) != 20:
            raise ValueError("each group must contain exactly 20 questions")
        all_ids.extend(record["stable_id"] for record in chosen)
        for record in chosen:
            if record["rank_sha256"] != rank(group, record["stable_id"]):
                raise ValueError("selection rank SHA mismatch")
        if group.startswith("GPQA-"):
            eligible = [row for row in raw[group] if bool(row["is_diamond"]) == (group == "GPQA-Diamond")]
            candidates = [{"stable_id": row["question_id"], "source_index": row["canonical_main_index"]} for row in eligible]
        elif group == "AIME24":
            candidates = [{"stable_id": f"aime24:{row['id']}", "source_index": i} for i, row in enumerate(raw[group])]
        elif group == "HumanEval+":
            candidates = [{"stable_id": row["task_id"], "source_index": i} for i, row in enumerate(raw[group])]
        elif group == "MedQA":
            candidates = [{"stable_id": f"medqa:{i}:{row.get('idx', 'no-idx')}:{row.get('pid', 'no-pid')}", "source_index": i} for i, row in enumerate(raw[group])]
        else:
            candidates = [{"stable_id": f"mmlu_pro:{row['category']}:{row['question_id']}", "source_index": i, "category": row["category"]} for i, row in enumerate(raw[group])]
        if group == "MMLU-Pro":
            categories = sorted({record["category"] for record in candidates})
            if len(categories) != 14:
                raise ValueError("MMLU category source changed")
            expected = [min((record for record in candidates if record["category"] == category), key=lambda record: rank(group, record["stable_id"])) for category in categories]
            selected_ids = {record["stable_id"] for record in expected}
            expected += sorted((record for record in candidates if record["stable_id"] not in selected_ids), key=lambda record: rank(group, record["stable_id"]))[:6]
        else:
            expected = sorted(candidates, key=lambda record: rank(group, record["stable_id"]))[:20]
        if [{k: record[k] for k in ("stable_id", "source_index")} for record in expected] != [{k: record[k] for k in ("stable_id", "source_index")} for record in chosen]:
            raise ValueError(f"answer-blind selection does not reproduce: {group}")
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != 120:
        raise ValueError("selection IDs are not unique")


def build_item(group, selected, raw_row, index, group_index, tests):
    item = dict(index=index, global_index=index, group_index=group_index,
        dataset_group=group, question_id=selected["stable_id"],
        source_index=selected["source_index"], rank_sha256=selected["rank_sha256"],
        is_diamond=group == "GPQA-Diamond", category="", n_options=0)
    if group == "AIME24":
        answer = str(raw_row["answer"]).strip()
        if not answer.isdigit() or not 0 <= int(answer) <= 999:
            raise ValueError("invalid AIME integer gold")
        item.update(dataset="aime2024", task="aime2024", question=raw_row["problem"].strip(),
                    gold=str(int(answer)), category="olympiad_math", evaluation_type="integer_exact")
    elif group.startswith("GPQA-"):
        if raw_row["question_id"] != selected["stable_id"] or raw_row["canonical_main_index"] != selected["source_index"]:
            raise ValueError("GPQA source row identity mismatch")
        if bool(raw_row["is_diamond"]) != item["is_diamond"]:
            raise ValueError("GPQA stratum mismatch")
        item.update(dataset="gpqa", task="gpqa", question=raw_row["question"],
                    gold=str(raw_row["gold"]).strip().upper(), n_options=4,
                    category="diamond" if item["is_diamond"] else "non_diamond", evaluation_type="choice_exact")
    elif group == "HumanEval+":
        if raw_row["task_id"] != selected["stable_id"] or "def check(" not in raw_row["test"]:
            raise ValueError("HumanEval+ identity/test mismatch")
        entry = raw_row["entry_point"]
        question = f"Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:\n{raw_row['prompt']}\n"
        rendered = raw_row["test"].replace("candidate", entry) + f"\n\ncheck({entry})"
        tests[selected["stable_id"]] = dict(task_id=raw_row["task_id"], entry_point=entry,
            raw_test=raw_row["test"], raw_test_sha256=hashlib.sha256(raw_row["test"].encode("utf-8")).hexdigest(),
            rendered_test=rendered, rendered_test_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            canonical_solution=raw_row["canonical_solution"], prompt=raw_row["prompt"],
            source_index=selected["source_index"], scorer="LatentMAS_HF_HumanEvalPlus_tests")
        item.update(dataset="humanevalplus", task="humanevalplus", question=question,
                    gold=None, entry_point=entry, private_test_key=selected["stable_id"],
                    category="code", evaluation_type="isolated_unit_tests")
    elif group == "MedQA":
        if len(raw_row["options"]) != 4:
            raise ValueError("MedQA source must have 4 choices")
        matches = [i for i, option in enumerate(raw_row["options"]) if str(raw_row["answer"]) in str(option)]
        if len(matches) != 1:
            raise ValueError("MedQA gold option is not uniquely matched")
        item.update(dataset="medqa", task="medqa", question=raw_row["query"],
                    gold=chr(65+matches[0]), n_options=4, category="medicine", evaluation_type="choice_exact")
    elif group == "MMLU-Pro":
        options = list(raw_row["options"])
        if not 2 <= len(options) <= 10 or raw_row["answer"] != chr(65+raw_row["answer_index"]):
            raise ValueError("invalid MMLU option schema")
        question = raw_row["question"].strip() + "\n\nChoices:\n" + "\n".join(f"({chr(65+i)}) {option}" for i, option in enumerate(options))
        item.update(dataset="mmlu_pro", task="gpqa", question=question, gold=raw_row["answer"],
                    n_options=len(options), category=raw_row["category"], evaluation_type="choice_exact")
    else:
        raise ValueError("unsupported group")
    item["question_sha256"] = hashlib.sha256(item["question"].encode("utf-8")).hexdigest()
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if sha(SELECTION) != SELECTION_SHA:
        raise ValueError("selection manifest SHA changed")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    dependencies, sources, raw = {}, {}, {}
    dependencies["dependencies/selection_manifest.json"] = copy_exact(SELECTION, output / "dependencies/selection_manifest.json", SELECTION_SHA)
    for group in GROUPS:
        source = selection["sources"][group]
        path = Path(source["path"])
        relative = SOURCE_TARGETS[group]
        dependencies[relative] = copy_exact(path, output / relative, source["file_sha256"])
        raw[group] = load_rows(path)
        source_count = sum(bool(row["is_diamond"]) == (group == "GPQA-Diamond") for row in raw[group]) if group.startswith("GPQA-") else len(raw[group])
        if source_count != source["rows"]:
            raise ValueError("source population count mismatch")
        sources[group] = dict(source, path=relative, selected=20, population=source_count)
    verify_selection(selection, raw)
    for name, digest in EXPECTED_AUTHOR.items():
        relative = "dependencies/LatentMAS/" + name
        dependencies[relative] = copy_exact(AUTHOR / name, output / relative, digest)
    dependencies["dependencies/LatentMAS/LICENSE"] = copy_exact(AUTHOR / "LICENSE", output / "dependencies/LatentMAS/LICENSE")
    for name, digest in EXPECTED_SANDBOX.items():
        relative = "dependencies/scoring/" + name
        dependencies[relative] = copy_exact(SANDBOX / name, output / relative, digest)
    questions, tests = [], {}
    for group in GROUPS:
        for group_index, selected in enumerate(selection["selected"][group]):
            questions.append(build_item(group, selected, raw[group][selected["source_index"]], len(questions), group_index, tests))
    if len(questions) != 120 or len({row["question_id"] for row in questions}) != 120:
        raise ValueError("question identity mismatch")
    write_json(output / "questions.json", questions)
    test_path = output / "dependencies/humanevalplus_tests.json"
    write_json(test_path, tests)
    dependencies["dependencies/humanevalplus_tests.json"] = {"sha256": sha(test_path), "bytes": test_path.stat().st_size}
    engineering_indices = [next(row["index"] for row in questions if row["dataset_group"] == group) for group in GROUPS]
    manifest = dict(passed=True, prepared_only=True, model_forward_calls=0,
        selection_manifest_sha256=SELECTION_SHA, salt=SALT,
        selection_rule="sha256(salt|dataset_group|stable_id), ascending; MMLU14 categories + global6",
        groups={group:20 for group in GROUPS}, total_questions=120, methods=list(METHODS), target_keys=1440,
        counts=dict(Counter(row["dataset"] for row in questions)), queue_batch_size=30, queue_batches=4,
        engineering_indices=engineering_indices, engineering_keys=72, engineering_reused_after_gate=True,
        questions_sha256=sha(output / "questions.json"), questions_bytes=(output / "questions.json").stat().st_size,
        independent_statistics_by_group=True, sources=sources, dependencies=dependencies,
        upstream_latentmas_commit=AUTHOR_COMMIT,
        humaneval_scorer=dict(mode="LatentMAS_paper_HF_test_script", timeout_seconds=10,
            evalplus_cli_used=False, tests_private=True, canonical_solution_never_prompted=True,
            isolated_executor="dependencies/scoring/sandbox.py", platform="Linux Landlock + seccomp; cloud gate required"),
        private_inputs_never_commit=True)
    write_json(output / "data_manifest.json", manifest)
    lock = dict(adapter_version="role_geometry_run_20261110_v1", selection_manifest_sha256=SELECTION_SHA,
        source_files={name:sha(HERE/name) for name in ("task_adapter.py", "prepare_data.py", "test_task_adapter.py")},
        upstream=dict(path=str(AUTHOR), commit=AUTHOR_COMMIT, files=EXPECTED_AUTHOR),
        isolated_executor=dict(path=str(SANDBOX), files=EXPECTED_SANDBOX),
        private_artifacts={"questions.json":sha(output/"questions.json"), "data_manifest.json":sha(output/"data_manifest.json"),
                           "dependencies/humanevalplus_tests.json":sha(test_path)},
        model_forward_calls=0, source_only_lock=True)
    write_json(HERE / "adapter_source_lock.json", lock)
    print(json.dumps(dict(passed=True, groups=manifest["groups"], questions=120, target_keys=1440,
        engineering_indices=engineering_indices, private_tests=len(tests),
        selection_sha256=SELECTION_SHA, questions_sha256=manifest["questions_sha256"],
        data_manifest_sha256=sha(output/"data_manifest.json"), model_forward_calls=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
