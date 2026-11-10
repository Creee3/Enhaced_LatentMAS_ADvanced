"""Final-only independent six-group scoring after complete coverage and audit."""
from __future__ import annotations
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time

from common import HERE, ROOT, load, save, records, verify, sha, identity
from task_adapter import normalized_gold, parse_answer, score_prediction
from sandbox_preflight import load_executor

METHODS = ("Full", "Gen", "H", "L", "H-OBF", "L-OBF", "G", "T", "R", "R-shuffle", "Pseudo-R", "Random-matched")
GROUPS = ("AIME24", "GPQA-Diamond", "GPQA-nonDiamond", "HumanEval+", "MedQA", "MMLU-Pro")


def result_key(row):
    return (row["dataset"], row["question_id"], int(row["latent_steps"]), row["method"])


def require_exact_complete(items, output, methods=METHODS):
    if len(items) != 120 or len({item["question_id"] for item in items}) != 120:
        raise ValueError("final score requires the fixed120 unique questions")
    if Counter(item["dataset_group"] for item in items) != Counter({group:20 for group in GROUPS}):
        raise ValueError("final score requires six independent20-question groups")
    expected = {(item["dataset"],item["question_id"],10,method) for item in items for method in methods}
    keys = [result_key(row) for row in output]
    if len(keys) != 1440 or len(set(keys)) != 1440 or set(keys) != expected:
        raise ValueError("final score requires1440 exact unique keys; no interim scoring")
    for row in output:
        item = items[int(row["index"])]
        if row["seed"] != 42 or row.get("status") != "ok":
            raise ValueError("result protocol/status mismatch")
        for name in ("index","global_index","dataset","dataset_group","question_id","is_diamond","category"):
            if row[name] != item[name]:
                raise ValueError("result/question identity mismatch")
    return {result_key(row):row for row in output}


def require_audit(audit, source_sha):
    if not audit.get("passed") or audit.get("accuracy_computed"):
        raise ValueError("successful pre-scoring structural audit is required")
    if audit.get("phase") not in ("formal","final") or audit.get("keys") != 1440 or audit.get("cases") != 120:
        raise ValueError("full final audit scope mismatch")
    if audit.get("own_reserved") != 0 or not audit.get("all_owned_services_stopped"):
        raise ValueError("final audit must certify stopped services and zero own reservations")
    if audit.get("run_lock_sha256") != source_sha:
        raise ValueError("final audit source SHA mismatch")


def exact_owner_dead(owner, lookup):
    actual = lookup(int(owner["pid"]))
    if actual is None:
        return True
    # PID reuse must not be confused with the original process.
    if actual.get("start_ticks") != owner.get("start_ticks"):
        return True
    if actual.get("uid") != owner.get("uid"):
        return True
    return False


def require_terminal_state(run_root, budget, lookup=identity):
    run_root = Path(run_root)
    queue = load(run_root/"queue.json")
    tasks = queue["tasks"]
    if len(tasks) != 120 or any(task["state"] != "complete" or task["methods"] for task in tasks):
        raise ValueError("all120 queue tasks must be complete with no missing methods")
    own = [reservation for reservation in budget["active_reservations"].values()
           if reservation.get("metadata",{}).get("experiment") == run_root.name]
    if own:
        raise ValueError("own reservations remain active")
    folder = run_root/"management"
    owner = load(folder/"identity.json")["identity"]
    if not exact_owner_dead(owner, lookup):
        raise ValueError("manager is still alive")
    if not load(folder/"finished.json").get("all_complete"):
        raise ValueError("manager has not recorded full completion")
    workers = load(folder/"workers.json")
    for worker in workers:
        if not exact_owner_dead(worker["identity"], lookup):
            raise ValueError("worker is still alive")
    stops = list(folder.glob("mps_*_stop.json"))
    if not stops or any(load(path).get("returncode") != 0 or load(path).get("owned_after") for path in stops):
        raise ValueError("owned MPS cleanup has not passed")
    return dict(manager_dead=True, workers_dead=len(workers), own_reserved=0, all_queue_complete=True)


def require_executor_preflight(run_root):
    run_root = Path(run_root)
    result = load(run_root/"executor_preflight.json")
    failures=result.get("known_canonical_numeric_failures",[])
    canonical_ok=result.get("canonical_passed")==20 and not failures and result.get("independent_witness_passed")==0
    declared_ok=result.get("canonical_passed")==19 and failures==["HumanEval/32"] and result.get("independent_witness_passed")==1
    if not result.get("passed") or result.get("model_forward_calls") != 0 or result.get("positive_gate_passed") !=20 or not (canonical_ok or declared_ok) or result.get("wrong_script_rejected") != 20 or not result.get("parser_negative_passed") or result.get("infrastructure_failures") != 0:
        raise ValueError("all20 positive/negative executor gates required; canonical numeric failure must be explicit")
    if result.get("questions_sha256") != sha(run_root/"questions.json") or result.get("tests_sha256") != sha(run_root/"dependencies/humanevalplus_tests.json"):
        raise ValueError("executor preflight belongs to different private inputs")
    if result.get("source_sha256") != sha(run_root/"sandbox_preflight.py"):
        raise ValueError("executor preflight source changed")
    if result.get("evaluator_overlay_lock_sha256")!=sha(run_root/"evaluator_overlay_lock.json") or result.get("evaluator_overlay_source_sha256")!=sha(run_root/"evaluator_overlay.py"):
        raise ValueError("explicit evaluator overlay evidence changed")
    from evaluator_overlay import verify_lock
    overlay_lock=verify_lock(run_root)
    if result.get("witness_sha256")!=overlay_lock["witness"]["sha256"]:
        raise ValueError("independent positive witness changed")
    manifest = load(run_root/"data_manifest.json")
    for relative,expected in result["dependencies"].items():
        if expected != manifest["dependencies"][relative] or sha(run_root/relative) != expected["sha256"]:
            raise ValueError("executor preflight dependency changed")
    return result


def numeric_summary(values):
    values = [float(value) for value in values]
    if not values:
        return None
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("timing metric must be finite and nonnegative")
    ordered = sorted(values)
    def percentile(fraction):
        point = (len(ordered)-1)*fraction
        left = int(point)
        right = min(left+1,len(ordered)-1)
        return ordered[left]+(ordered[right]-ordered[left])*(point-left)
    return dict(count=len(values), total_seconds=sum(values), mean_seconds=statistics.fmean(values),
                median_seconds=statistics.median(values), p10_seconds=percentile(.1), p90_seconds=percentile(.9),
                min_seconds=min(values), max_seconds=max(values))


def timing_components(row, run_root=None):
    timing = row["timing"]
    components = timing["components"]
    if len(components) != 3:
        raise ValueError("three handoff timing components required")
    # Native elapsed receipts include native prompt prefill+latent or final decode.
    # First native forward hooks additionally separate prefill from subsequent decoding; user-visible TTFT is not directly measured.
    stages = row["prefix_stages"]
    if len(stages) != 3:
        raise ValueError("three native latent receipts required")
    native = sum(float(stage["elapsed_seconds"]) for stage in stages) + float(row["judger_receipt"]["elapsed_seconds"])
    probes = float(timing["shared_first_probe_seconds"]) + sum(float(component["required_probe_seconds"]) for component in components)
    geometry = sum(float(component["geometry_seconds"]) for component in components)
    selector = sum(float(component["selector_and_gather_seconds"]) for component in components)
    common_probe = 0.0
    common_probe_count = 0
    if row["method"] == "Full" and run_root is not None:
        import numpy as np
        attempt = Path(row["_attempt_directory"])
        for receipt in row.get("common_state_diagnostic_files",[]):
            path = attempt/receipt["file"]
            digest = receipt.get("sha256",receipt.get("file_sha256"))
            if not digest or sha(path) != digest:
                raise ValueError("diagnostic timing source SHA mismatch")
            with np.load(path,allow_pickle=False) as stored:
                metadata = json.loads(str(stored["metadata_json"].item()))
            if "receiver_probe_seconds" in metadata:
                common_probe += float(metadata["receiver_probe_seconds"])
                common_probe_count += 1
    result = dict(native_model_generation_seconds=native, required_probe_seconds=probes,
        geometry_seconds=geometry, selector_and_gather_seconds=selector,
        reconstructed_instrumented_seconds=float(timing["reconstructed_standalone_seconds"]),
        arm_instrumented_seconds=float(timing["arm_instrumented_seconds"]),
        common_state_receiver_probe_seconds=common_probe, common_state_receiver_probe_calls=common_probe_count,
        native_prefill_seconds=float(stages[0].get('native_prefill_seconds',0))+sum(float(x.get('receiver_prefill_seconds',0)) for x in components)+float(timing.get('judger_prefill_seconds',0)),
        judger_prefill_seconds=float(timing.get('judger_prefill_seconds',0)),
        judger_decode_seconds=float(timing.get('judger_decode_seconds',0)),
        measured_component_sum_seconds=native+probes+geometry+selector+common_probe)
    for key,value in result.items():
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError("negative/non-finite timing")
    return result


def summarize_group(items, by_key, scored, methods=METHODS, run_root=None):
    if len(items) != 20:
        raise ValueError("group summary requires20 independent questions")
    result = {}
    for method in methods:
        selected = [by_key[(item["dataset"],item["question_id"],10,method)] for item in items]
        judgments = [scored[(item["dataset"],item["question_id"],10,method)] for item in items]
        actual = sum(int(sum(row["actual_kv_bytes_by_handoff"])) for row in selected)
        full = sum(sum(int(transfer["full_bytes_from_original_positions"]) for transfer in row["transfers"]) for row in selected)
        if full <= 0:
            raise ValueError("invalid full-cache byte denominator")
        details = [timing_components(row,run_root) for row in selected]
        timer_names = ("native_model_generation_seconds","required_probe_seconds","geometry_seconds",
            "selector_and_gather_seconds","reconstructed_instrumented_seconds","arm_instrumented_seconds",
            "common_state_receiver_probe_seconds","native_prefill_seconds","judger_prefill_seconds","judger_decode_seconds","measured_component_sum_seconds")
        result[method] = dict(total=20, correct=sum(judgment["passed"] for judgment in judgments),
            accuracy=sum(judgment["passed"] for judgment in judgments)/20,
            parse_valid=sum(row["parse_valid"] for row in selected), parse_failures=sum(not row["parse_valid"] for row in selected),
            actual_KV_bytes_by_handoff=[sum(int(row["actual_kv_bytes_by_handoff"][hop]) for row in selected) for hop in range(3)],
            cumulative_KV_bytes=actual, full_reference_cumulative_KV_bytes=full,
            cumulative_KV_retained_ratio=actual/full, cumulative_KV_reduction_ratio=1-actual/full,
            cap_hits=sum(row["generated_tokens"]==8192 for row in selected),
            timing={name:numeric_summary([detail[name] for detail in details]) for name in timer_names},
            common_state_receiver_probe_calls=sum(detail["common_state_receiver_probe_calls"] for detail in details))
    return result


def build_summary(items, output, run_root, executor, tests):
    by_key = require_exact_complete(items,output)
    scored = {}
    private_scores = []
    for item in items:
        for method in METHODS:
            key = (item["dataset"],item["question_id"],10,method)
            row = by_key[key]
            prediction = parse_answer(item,row["answer_text"])
            if prediction != row["prediction"] or bool(row["parse_valid"]) != (prediction is not None):
                raise ValueError("saved answer parse does not reproduce")
            judgment = score_prediction(item,prediction,tests,executor,10)
            if judgment.get("error_type") == "SandboxFailure":
                raise ValueError("isolated scorer infrastructure failure")
            judgment["passed"] = bool(judgment.get("passed"))
            scored[key] = judgment
            private_scores.append(dict(dataset_group=item["dataset_group"],question_id=item["question_id"],method=method,
                parse_valid=prediction is not None,passed=judgment["passed"],error_type=judgment.get("error_type"),scorer=judgment["scorer"]))
    groups = {}
    for group in GROUPS:
        selected = [item for item in items if item["dataset_group"] == group]
        groups[group] = dict(n=20,methods=summarize_group(selected,by_key,scored,run_root=run_root))
    return dict(complete=True,questions=120,keys=1440,methods=list(METHODS),datasets=groups,
        no_pooled_accuracy=True,statistical_unit="question within dataset_group",exploratory=True,
        timing_scope="native receipts + required probes + geometry + selector; per-method reconstructed standalone includes shared Planner/probe once; not independently timed throughput",
        common_state_probe_scope="diagnostic overhead attributed to Full loop only; not intrinsic Full runtime",
        ttft_breakdown_available=False,native_prefill_decode_breakdown_available=True,accuracy_computed_after_final_audit=True), private_scores


def main():
    started = time.time()
    source = verify()
    manifest = load(HERE/"data_manifest.json")
    if sha(HERE/"questions.json") != manifest["questions_sha256"]:
        raise ValueError("question SHA mismatch")
    plan = load(HERE/"plan.json")
    if plan["methods"] != list(METHODS) or plan["target_keys"] != 1440:
        raise ValueError("score method/scope mismatch")
    audit = load(HERE/"formal_audit_verified.json")
    require_audit(audit,source["run_lock_sha256"])
    # The raw complete archive precedes any correctness calculation.
    archive = load(HERE/"formal_archive_verified.json")
    if not archive.get("passed"):
        raise ValueError("verified full raw archive required before scoring")
    terminal = require_terminal_state(HERE,load(ROOT/"cumulative_budget.json"))
    executor_preflight = require_executor_preflight(HERE)
    executor = load_executor(HERE)
    tests = load(HERE/"dependencies/humanevalplus_tests.json")
    output = records()
    # Attach private file provenance without altering originals.
    origin = {}
    for attempt in (HERE/"attempts").iterdir():
        result_file = attempt/"results.jsonl"
        if not result_file.exists():
            continue
        from common import rows
        for row in rows(result_file):
            key = result_key(row)
            if key in origin:
                raise ValueError("duplicate source result key")
            origin[key] = str(attempt)
    output = [dict(row,_attempt_directory=origin[result_key(row)]) for row in output]
    # Check complete coverage before executing any test or calculating accuracy.
    require_exact_complete(load(HERE/"questions.json"),output)
    summary,scores = build_summary(load(HERE/"questions.json"),output,HERE,executor,tests)
    summary.update(run_lock_sha256=source["run_lock_sha256"],data_manifest_sha256=sha(HERE/"data_manifest.json"),
        terminal_state=terminal,structural_audit_sha256=sha(HERE/"formal_audit_verified.json"),
        raw_archive_proof_sha256=sha(HERE/"formal_archive_verified.json"),executor_preflight_sha256=sha(HERE/"executor_preflight.json"),
        scoring_model_forward_calls=0,scoring_started_unix=started,scoring_finished_unix=time.time())
    save(HERE/"final_scores.json",dict(complete=True,keys=1440,scores=scores,model_forward_calls=0))
    save(HERE/"final_summary.json",summary)
    report=["# Role geometry120 final six-group report","", "All1440 method-question keys are complete. Each dataset group is reported independently; no pooled accuracy is computed.","", "| Dataset group | "+" | ".join(METHODS)+" |", "|---|"+"---:|"*len(METHODS)]
    for group in GROUPS:
        values=summary["datasets"][group]["methods"]
        report.append("| "+group+" | "+" | ".join(f"{values[method]['correct']}/20 ({values[method]['accuracy']:.1%})" for method in METHODS)+" |")
    report += ["", "KV ratios use cumulative native tensor bytes divided by the Full bytes of each method's original position history. They are not wall-time speedups.",
        "Native generation, required diagnostic probes, geometry and selector/gather times are separate in final_summary.json. Shared prefix costs are charged once per reconstructed method; these are not independently timed standalone jobs.",
        "The worker records native first-prefill and remaining latent/decode time; user-visible TTFT is not directly measured. Full-loop common-state diagnostic overhead is explicitly separate.",
        "HumanEval+ uses locked LatentMAS/HF test scripts in the isolated executor; no EvalPlus CLI claim is made."]
    (HERE/"final_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps(dict(complete=True,questions=120,keys=1440,groups=6,summary_sha256=sha(HERE/"final_summary.json"),scoring_model_forward_calls=0)))


if __name__ == "__main__":
    main()
