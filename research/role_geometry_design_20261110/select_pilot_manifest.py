"""Create a private, answer-blind SHA-ranked role-geometry pilot manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(r"D:/liuxin_workspace/ICML2027")
OUT = Path(__file__).resolve().parent
SALT = "role_geometry_pilot_20261110_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(group: str, stable_id: str) -> str:
    return hashlib.sha256(f"{SALT}|{group}|{stable_id}".encode()).hexdigest()


def choose(group: str, rows: list[dict], n: int) -> list[dict]:
    ranked = sorted(rows, key=lambda r: rank(group, r["stable_id"]))
    return [dict(r, rank_sha256=rank(group, r["stable_id"])) for r in ranked[:n]]


def json_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    gpqa_path = ROOT / r"advanced_MOC/experiments/latentmas_role_blocks_main448_20261102/cohort_questions.json"
    aime_path = ROOT / r"课程材料/LatentMAS_GPU2_复现_20261107/downloaded_data/HuggingFaceH4--aime_2024--data--train-00000-of-00001.parquet"
    aime_manifest = ROOT / r"课程材料/LatentMAS_GPU2_复现_20261107/downloaded_data/HuggingFaceH4--aime_2024.json"
    human_path = ROOT / r"课程材料/LatentMAS_GPU2_复现_20261107/downloaded_data/evalplus--humanevalplus--data--test-00000-of-00001-5973903632b82d40.parquet"
    human_manifest = ROOT / r"课程材料/LatentMAS_GPU2_复现_20261107/downloaded_data/evalplus--humanevalplus.json"
    medqa_path = ROOT / r"04_参考代码与历史实验/上游参考代码/LatentMAS-official/data/medqa.json"
    mmlu_path = ROOT / r"advanced_MOC/experiments/cloud_completed_backup_20261110/runs/HL_705exp_s42_l10_20261109T124902Z/source_data/TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet"

    gpqa = json_rows(gpqa_path)
    groups: dict[str, list[dict]] = {
        "AIME24": [
            {"stable_id": f"aime24:{r['id']}", "source_index": i}
            for i, r in enumerate(pq.read_table(aime_path).to_pylist())
        ],
        "GPQA-Diamond": [
            {"stable_id": r["question_id"], "source_index": r["canonical_main_index"]}
            for r in gpqa if r["is_diamond"]
        ],
        "GPQA-nonDiamond": [
            {"stable_id": r["question_id"], "source_index": r["canonical_main_index"]}
            for r in gpqa if not r["is_diamond"]
        ],
        "HumanEval+": [
            {"stable_id": r["task_id"], "source_index": i}
            for i, r in enumerate(pq.read_table(human_path).to_pylist())
        ],
        "MedQA": [
            {"stable_id": f"medqa:{i}:{r.get('idx', 'no-idx')}:{r.get('pid', 'no-pid')}", "source_index": i}
            for i, r in enumerate(json_rows(medqa_path))
        ],
    }

    mmlu = pq.read_table(mmlu_path).to_pylist()
    by_category: dict[str, list[dict]] = {}
    for i, r in enumerate(mmlu):
        category = r["category"]
        by_category.setdefault(category, []).append(
            {"stable_id": f"mmlu_pro:{category}:{r['question_id']}", "source_index": i, "category": category}
        )
    # One question per category, then six additional globally lowest-ranked rows.
    selected_mmlu: list[dict] = []
    for category in sorted(by_category):
        selected_mmlu.append(min(by_category[category], key=lambda r: rank("MMLU-Pro", r["stable_id"])))
    selected_ids = {r["stable_id"] for r in selected_mmlu}
    remaining = [r for rows in by_category.values() for r in rows if r["stable_id"] not in selected_ids]
    selected_mmlu.extend(sorted(remaining, key=lambda r: rank("MMLU-Pro", r["stable_id"]))[: max(0, 20 - len(selected_mmlu))])
    groups["MMLU-Pro"] = selected_mmlu

    selected = {}
    for group, rows in groups.items():
        if group == "MMLU-Pro":
            selected[group] = [dict(r, rank_sha256=rank(group, r["stable_id"])) for r in rows]
        else:
            selected[group] = choose(group, rows, 20)

    sources = {
        "AIME24": {"path": str(aime_path), "file_sha256": sha256(aime_path), "manifest_sha256": sha256(aime_manifest), "rows": len(groups["AIME24"])},
        "GPQA-Diamond": {"path": str(gpqa_path), "file_sha256": sha256(gpqa_path), "rows": len(groups["GPQA-Diamond"])},
        "GPQA-nonDiamond": {"path": str(gpqa_path), "file_sha256": sha256(gpqa_path), "rows": len(groups["GPQA-nonDiamond"])},
        "HumanEval+": {"path": str(human_path), "file_sha256": sha256(human_path), "manifest_sha256": sha256(human_manifest), "rows": len(groups["HumanEval+"])},
        "MedQA": {"path": str(medqa_path), "file_sha256": sha256(medqa_path), "rows": len(groups["MedQA"])},
        "MMLU-Pro": {"path": str(mmlu_path), "file_sha256": sha256(mmlu_path), "rows": len(mmlu), "categories": sorted(by_category)},
    }
    output = {
        "status": "prepared_not_run",
        "purpose": "exploratory role-geometry/readout pilot; no accuracy claim",
        "salt": SALT,
        "hash_rule": "sha256(salt|dataset_group|stable_id), ascending",
        "groups": {k: 20 for k in selected},
        "total_questions": sum(len(v) for v in selected.values()),
        "sources": sources,
        "selected": selected,
        "answer_blind_selection": True,
        "gsm8k_included": False,
    }
    (OUT / "pilot_selection_manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT / "pilot_selection_manifest.json"), "groups": output["groups"], "total": output["total_questions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
