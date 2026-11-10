"""Independent live-worker mechanics certificate; no model/SSH/score actions.

Only complete JSONL lines are inspected. Healthy in-flight requests are allowed;
all seven RIDs backing a committed answer must already have raw+settlement
records. Nothing settles or edits the shared ledger, queue, worker or model.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common import HERE, ROOT, SCIENCE, BASE, METHODS, load, save, sha, identity, verify


def stable_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    # The last incomplete line can be an active append; ignore it, never repair.
    return [json.loads(line) for line in path.read_bytes().split(b"\n")[:-1] if line.strip()]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expected_native_ast():
    tree = ast.parse((SCIENCE/"cloud_worker.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == "main")
    names = ("timed", "context", "stage", "judge")
    functions = [node for node in ast.walk(main) if isinstance(node,ast.FunctionDef) and node.name in names]
    require(len(functions) == 4, "frozen native functions missing")
    digest = hashlib.sha256(ast.dump(ast.Module(body=functions,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    return names, digest


def safe_file(out, relative):
    path = (out/relative).resolve()
    require(path.is_relative_to(out.resolve()), "artifact path outside worker attempt")
    require(path.is_file(), "committed artifact missing")
    return path


def grouped_unique(records, event):
    selected = [record for record in records if record.get("event") == event]
    counts = Counter(record["reservation_id"] for record in selected)
    require(all(count == 1 for count in counts.values()), event+" contains duplicate RIDs")
    return {record["reservation_id"]:record for record in selected}


def check_diagnostic(out, receipt, row, common, seen):
    path = safe_file(out, receipt["file"])
    if str(path) in seen:
        return
    require(sha(path) == receipt["sha256"], "diagnostic SHA mismatch")
    require(receipt.get("readback_verified") and not receipt.get("raw_kv_retained"), "diagnostic compact/readback proof missing")
    with np.load(path,allow_pickle=False) as pack:
        require("metadata_json" in pack.files, "diagnostic metadata missing")
        meta = json.loads(str(pack["metadata_json"].item()))
        require(1 <= int(meta["hop"]) <= 3, "diagnostic handoff outside1..3")
        if common:
            require(meta.get("analysis") == "same_full_state" and meta["method"] in METHODS, "common-state diagnostic identity mismatch")
        else:
            require(meta.get("method") == row["method"], "own diagnostic method mismatch")
        if "source_cache_sha" in meta:
            require(len(meta["source_cache_sha"]) == 64, "source cache SHA missing")
        for name in pack.files:
            value = pack[name]
            require(value.dtype.kind != "O", "object payload in diagnostic")
            require(name not in {"raw_k","raw_v","attention","tokens","key","value","keys","values"}, "raw tensor in compact diagnostic")
            if value.dtype.kind in "fc":
                require(bool(np.isfinite(value).all()), "nonfinite diagnostic array")
    seen.add(str(path))


def check_row(row, item, out, source, native_ast, raw, started, settled, seen):
    method = row["method"]
    require(method in METHODS and row["seed"] == 42 and row["latent_steps"] == 10, "method/seed/latent scope mismatch")
    require(all(row[field] == item[field] for field in ("dataset","dataset_group","question_id","category","is_diamond")), "result dataset identity mismatch")
    require(row["source_lock_sha"] == source["run_lock_sha256"] and row["native_functions_ast_sha"] == native_ast, "result source/native AST mismatch")
    rids = row["request_ids"]
    require(len(rids) == len(set(rids)) == 7 and len(row["shared_request_ids"]) == 2, "answer must reference seven unique raw RIDs")
    require(set(row["shared_request_ids"]).issubset(rids), "shared RIDs missing from answer")
    for rid in rids:
        require(rid in raw and rid in started and rid in settled, "committed RID missing raw/start/settlement evidence")
        value, charge = raw[rid], settled[rid]
        require(value.get("question_id") == item["question_id"] and value.get("dataset") == item["dataset"], "raw RID has different question identity")
        require(not value["failed"] and not charge["failed"] and charge["independent_unknown"] == 0, "committed answer references failed/unknown request")
        require(value["actual_positions"] == charge["actual_positions"] and value["native_forward_calls"] == charge["forward_calls"], "raw/settled position cost mismatch")
    judge = row["judger_receipt"]
    require(judge["reservation_id"] in rids and raw[judge["reservation_id"]]["generated_ids"] == judge["generated_ids"], "judger token/raw RID mismatch")
    require(hashlib.sha256(json.dumps(judge["generated_ids"]).encode()).hexdigest() == row["token_sha"], "judger token SHA mismatch")
    require(0 < row["generated_tokens"] == len(judge["generated_ids"]) <= 8192, "judger token count mismatch")
    proof = judge["position_proof"]
    require(proof["positions"] == judge["actual_positions"] and proof["sdpa_checks"] == 36*proof["forwards"], "native judger position/RoPE proof mismatch")
    require(len(row["prefix_stages"]) == len(row["transfers"]) == len(row["position_files"]) == len(row["receiver_fidelity"]) == 3, "three handoff evidence missing")
    require(len(row["diagnostic_files"]) == len({entry["file"] for entry in row["diagnostic_files"]}) == 9, "nine distinct per-answer diagnostic files required")
    previous_positions = None
    for hop,(transfer,receipt,stage) in enumerate(zip(row["transfers"],row["position_files"],row["prefix_stages"]),1):
        require(transfer["handoff"] == hop and transfer["method"] == method, "transfer method/hop mismatch")
        require(transfer["source_k_equal_measured"] and transfer["protected_latent_sink_equal_measured"] and transfer["original_RoPE"], "original KV/RoPE protection failed")
        if method in ("H-OBF","L-OBF"):
            rank = 2 if method == "H-OBF" else 4
            require(transfer.get("pca_rank",transfer.get("obf_rank")) == rank, "OBF paper rank mismatch")
            require(transfer.get("protected_v_equal_measured") and transfer.get("history_preserved_equal_measured"), "OBF protected V/history check missing")
        else:
            require(transfer["source_v_equal_measured"], "eviction-only V changed")
        path = safe_file(out,receipt["file"])
        require(sha(path) == receipt["file_sha256"], "position file SHA mismatch")
        with np.load(path,allow_pickle=False) as pack:
            pos,indices = pack["positions"],pack["storage_indices"]
            retained = int(transfer["retained"])
            require(pos.shape == indices.shape == (36,8,retained), "position tensor shape not36x8xN")
            require(pos.dtype.kind in "iu" and indices.dtype.kind in "iu", "positions/indices must be integer")
            require(bool((pos >= 0).all() and (pos < stage["original_end"]).all() and (np.diff(pos,axis=-1) > 0).all()), "original positions out of range/order")
            require(hashlib.sha256(pos.tobytes()).hexdigest() == receipt["position_sha"] == transfer["position_sha"], "original position SHA mismatch")
            fresh = np.arange(stage["original_before"],stage["original_end"],dtype=pos.dtype)
            allmap = np.broadcast_to(fresh,(36,8,len(fresh))) if previous_positions is None else np.concatenate((previous_positions,np.broadcast_to(fresh,(36,8,len(fresh)))),axis=-1)
            require(bool((indices >= 0).all() and (indices < allmap.shape[-1]).all()), "storage indices out of range")
            require(np.array_equal(np.take_along_axis(allmap,indices,axis=-1),pos), "original-position gather mismatch")
            require(bool(((pos < 4).sum(-1) == 4).all()), "sink protection failed")
            for earlier in row["prefix_stages"][:hop]:
                require(bool((((pos >= earlier["original_end"]-10)&(pos < earlier["original_end"])).sum(-1) == 10).all()), "latent protection failed")
            if method == "Full":
                require(bool((pos == np.arange(stage["original_end"])).all()), "Full deletion detected")
            elif method == "Gen":
                require(retained == 4+10*hop, "Gen retained count mismatch")
            elif method in ("H","L","H-OBF","L-OBF"):
                previous = 0 if previous_positions is None else previous_positions.shape[-1]
                sink = 4 if hop == 1 else 0
                require(retained == previous+sink+min(32,stage["prompt_len"]-sink)+10, "author prompt budget mismatch")
                require(transfer.get("independent_author_selection_equal") and transfer.get("sender_attention_steps") == 10, "author selection equality missing")
                if method in ("L","L-OBF"):
                    require(bool((pos == pos[:,:1,:]).all()), "layerwise positions differ across heads")
            else:
                # Each prior relay uses <=96 prompt rows; its original prompt
                # lengths may be much larger than retained candidate count.
                actual_candidate = allmap.shape[-1]-4-10*hop
                require(retained == min(96,actual_candidate)+4+10*hop, "relay96 prompt budget mismatch")
                end = stage["original_before"]+stage["prompt_len"]
                lo = max(4,stage["original_before"],end-8)
                require(bool((((pos >= lo)&(pos < end)).sum(-1) == end-lo).all()), "mandatory recent rows dropped")
            require(transfer["actual_kv_bytes"] == retained*147456, "actual KV bytes mismatch")
            require(transfer["full_bytes_from_original_positions"] == stage["original_end"]*147456, "Full byte reference mismatch")
            previous_positions = pos.copy()
    require(row["actual_cumulative_relay_bytes"] == sum(transfer["actual_kv_bytes"] for transfer in row["transfers"]), "cumulative KV bytes mismatch")
    for receipt in row["diagnostic_files"]:
        check_diagnostic(out,receipt,row,False,seen)
    for receipt in row.get("common_state_diagnostic_files",[]):
        check_diagnostic(out,receipt,row,True,seen)


def check_worker(worker, source, native_names, native_ast, items):
    out = Path(worker["out"])
    actual = identity(worker["pid"])
    owner = worker["identity"]
    live = actual is not None and all(actual.get(field) == owner.get(field) for field in ("uid","start_ticks","argv","CUDA_VISIBLE_DEVICES"))
    if actual is not None and not live:
        raise AssertionError("PID identity changed; cannot claim current process")
    ready_path = out/"model_ready.json"
    if not ready_path.exists():
        return dict(pid=worker["pid"],uid=owner["uid"],start_ticks=owner["start_ticks"],gpu=worker["gpu_index"],live=live,status="loading_unverified",new_rows_by_method={},verified_new_rows=0,all_methods_verified=False,complete_12_method_cases=0)
    ready = load(ready_path)
    require(all(ready["identity"].get(field) == owner.get(field) for field in ("pid","uid","start_ticks","argv","CUDA_VISIBLE_DEVICES")), "model-ready identity mismatch")
    require(ready["source"]["run_lock_sha256"] == source["run_lock_sha256"], "worker model-ready source mismatch")
    require(ready["weight_files"] == 15 and ready["bf16_cuda"] and ready["parameter_bytes"] == 16381470720 and ready["identity_norm"] and abs(ready["target_norm"]-1.3758320808410645)<1e-6, "BF16 CUDA model/identity/norm mismatch")
    if not (out/"native_functions.json").exists() and not stable_rows(out/"results.jsonl"):
        return dict(pid=worker["pid"],uid=owner["uid"],start_ticks=owner["start_ticks"],gpu=worker["gpu_index"],live=live,status="loaded_native_verification_pending",new_rows_by_method={},verified_new_rows=0,all_methods_verified=False,complete_12_method_cases=0)
    native = load(out/"native_functions.json")
    require(set(native["names"]) == set(native_names) and native["ast_sha256"] == native_ast and native["source_file_sha256"] == sha(SCIENCE/"cloud_worker.py"), "native stage/judge AST provenance mismatch")
    trace = stable_rows(out/"trace.jsonl")
    require({x["file"] for x in trace if x.get("event") == "model_file_verified"} == set(load(BASE/"model_lock.json")["files"]), "15 model SHA checks missing")
    started = grouped_unique(trace,"request_started")
    settled = grouped_unique(trace,"request_settled")
    raw = grouped_unique(stable_rows(out/"raw/requests_raw.jsonl"),"request_output")
    for rid in set(started)|set(settled)|set(raw):
        require(rid.startswith(str(worker["pid"])+"-"), "RID owner PID mismatch")
    results = stable_rows(out/"results.jsonl")
    keys = [(row["index"],row["method"]) for row in results]
    require(len(keys) == len(set(keys)), "duplicate worker result keys")
    seen = set()
    for row in results:
        require(0 <= int(row["index"]) < len(items), "result index outside120")
        check_row(row,items[row["index"]],out,source,native_ast,raw,started,settled,seen)
    methods_by_case = defaultdict(set)
    for row in results:
        methods_by_case[row["index"]].add(row["method"])
    complete = sum(methods == set(METHODS) for methods in methods_by_case.values())
    for case,methods in methods_by_case.items():
        if methods != set(METHODS):
            continue
        group = [row for row in results if row["index"] == case]
        require(len({row["shared_planner_kv_sha"] for row in group}) == len({row["fork_rng_sha"] for row in group}) == len({tuple(row["shared_request_ids"]) for row in group}) == 1, "complete case methods do not share Planner/prefix RNG")
        require(len({rid for row in group for rid in row["request_ids"]}) == 62, "complete12-method answer RID union must be62")
        require(len(set(group[0].get("common_state_diagnostic_request_ids",[]))) == 22, "complete case common-state probe count must be22")
    counts = dict(Counter(row["method"] for row in results))
    return dict(pid=worker["pid"],uid=owner["uid"],start_ticks=owner["start_ticks"],gpu=worker["gpu_index"],gpu_uuid=worker["gpu_uuid"],live=live,status="12_methods_verified" if complete else ("partial_verified" if results else "ready_no_new_answers"),new_rows_by_method=counts,verified_new_rows=len(results),all_methods_verified=bool(complete),complete_12_method_cases=complete,weights_sha_verified=15,native_ast_verified=True,bf16_cuda_identity_norm=True,raw_RIDs_tokens_KV_RoPE_positions_bytes_verified=True,compact_diagnostics_verified=len(seen),started_requests=len(started),settled_requests=len(settled),inflight_request_count=len(set(started)-set(settled)),no_global_settlement_assumption=True)


def main(phase="engineering"):
    require(phase in ("engineering","formal"), "phase must be engineering or formal")
    source = verify()
    native_names,native_ast = expected_native_ast()
    workers_path = HERE/"management/workers.json"
    workers = load(workers_path) if workers_path.exists() else []
    items = load(HERE/"questions.json")
    checks,failures = [],[]
    for worker in workers:
        out = Path(worker["out"])
        if (out/"failure.json").exists() or (out/"manager_observed_exit.json").exists():
            failures.append(dict(pid=worker["pid"],gpu=worker["gpu_index"],kind="worker_failure_scene_present"))
        try:
            checks.append(check_worker(worker,source,native_names,native_ast,items))
        except Exception as error:
            failures.append(dict(pid=worker["pid"],gpu=worker["gpu_index"],kind="verification_failure",error_type=type(error).__name__,reason=str(error)[:250]))
    budget = load(ROOT/"cumulative_budget.json")
    owned = [value for value in budget["active_reservations"].values() if value.get("metadata",{}).get("experiment") == HERE.name]
    proof = dict(passed=not failures,phase=phase,source=source,workers=checks,verified_failures=failures,own_inflight_reservations=len(owned),healthy_inflight_not_settled_by_checker=True,accuracy_computed=False,model_forward_calls=0,unix=time.time())
    save(HERE/"management/workers_verified.json",proof)
    print(json.dumps(dict(passed=proof["passed"],phase=phase,workers=[{field:entry.get(field) for field in ("pid","uid","start_ticks","gpu","live","status","new_rows_by_method","verified_new_rows","complete_12_method_cases")} for entry in checks],verified_failures=failures,own_inflight_reservations=len(owned),model_forward_calls=0)))
    return proof


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "engineering")
