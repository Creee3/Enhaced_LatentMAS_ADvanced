import os
"""Unit tests for final-only scoring and private code-evaluator boundaries."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

import score
from common import save
from sandbox_preflight import run_preflight
from task_adapter import score_prediction

GROUPS=("AIME24","GPQA-Diamond","GPQA-nonDiamond","HumanEval+","MedQA","MMLU-Pro")
METHODS=score.METHODS
DATA=Path(os.environ["ROLE_GEOMETRY_DATA_ROOT"]) if "ROLE_GEOMETRY_DATA_ROOT" in os.environ else Path(__file__).resolve().parents[3]/"advanced_MOC/experiments/latentmas_role_geometry120_20261110"


def fixtures():
    dataset_for={"AIME24":"aime2024","GPQA-Diamond":"gpqa","GPQA-nonDiamond":"gpqa","HumanEval+":"humanevalplus","MedQA":"medqa","MMLU-Pro":"mmlu_pro"}
    items=[]; rows=[]
    for g,ds in dataset_for.items():
        for n in range(20):
            item=dict(index=len(items),global_index=len(items),dataset=ds,dataset_group=g,question_id=f"{g}-{n}",
                      is_diamond=g=="GPQA-Diamond",category=g,n_options=4 if ds in ("gpqa","medqa") else (10 if ds=="mmlu_pro" else 0),
                      question=f"Fixture question {g} {n}",task={"aime2024":"aime2024","gpqa":"gpqa","humanevalplus":"humanevalplus","medqa":"medqa","mmlu_pro":"gpqa"}[ds],gold=None if ds=="humanevalplus" else ("1" if ds=="aime2024" else "A"),entry_point="f" if ds=="humanevalplus" else None,private_test_key=f"{g}-{n}" if ds=="humanevalplus" else None)
            items.append(item)
            for method in METHODS:
                rows.append(dict(dataset=ds,question_id=item["question_id"],latent_steps=10,method=method,index=item["index"],seed=42,status="ok",global_index=item["global_index"],dataset_group=g,is_diamond=item["is_diamond"],category=g,
                    actual_cumulative_relay_bytes=100,actual_kv_bytes_by_handoff=[30,30,40],
                    transfers=[dict(full_bytes_from_original_positions=100) for _ in range(3)],
                    parse_valid=True,generated_tokens=128,
                    prefix_stages=[dict(elapsed_seconds=1) for _ in range(3)],
                    judger_receipt=dict(elapsed_seconds=2),
                    timing=dict(components=[dict(required_probe_seconds=1,geometry_seconds=2,selector_and_gather_seconds=3) for _ in range(3)],
                        shared_first_probe_seconds=1,reconstructed_standalone_seconds=12,arm_instrumented_seconds=10)))
    return items,rows


class FinalScoreTests(unittest.TestCase):
    def test_exact_coverage_is_required_before_scoring(self):
        items, rows=fixtures()
        result=score.require_exact_complete(items,rows)
        self.assertEqual(len(result),1440)
        with self.assertRaises(ValueError):
            score.require_exact_complete(items,rows[:-1])
        duplicate=rows+[copy.deepcopy(rows[0])]
        with self.assertRaises(ValueError):
            score.require_exact_complete(items,duplicate)
        rows[0]["seed"]=7
        with self.assertRaises(ValueError):
            score.require_exact_complete(items,rows)

    def test_accuracy_is_group_local(self):
        items,rows=fixtures()
        fake={}
        for row in rows:
            fake[(row["dataset"],row["question_id"],10,row["method"])]=row
        scored={key:{"passed":False} for key in fake}
        result=score.summarize_group(items[:20],fake,scored)
        self.assertEqual(set(result),set(METHODS))
        self.assertEqual({v["total"] for v in result.values()},{20})
        self.assertEqual({v["accuracy"] for v in result.values()},{0.0})
        with self.assertRaises(ValueError):
            score.summarize_group(items[:19],fake,scored)

    def test_terminal_gate_requires_all_own_services_dead(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"management").mkdir();(root/"attempts").mkdir()
            tasks=[dict(state="complete",methods=[]) for _ in range(120)]
            save(root/"queue.json",dict(tasks=tasks,engineering_indices=[0,20,40,60,80,100]))
            owner=dict(pid=123,uid=1000,start_ticks=456,argv=["manager"])
            save(root/"management/identity.json",dict(identity=owner))
            save(root/"management/finished.json",dict(all_complete=True))
            save(root/"management/workers.json",[])
            save(root/"management/mps_0_stop.json",dict(returncode=0,owned_after=[]))
            budget=dict(active_reservations={})
            self.assertTrue(score.require_terminal_state(root,budget,lookup=lambda pid:None)["manager_dead"])
            with self.assertRaises(ValueError):
                score.require_terminal_state(root,budget,lookup=lambda pid:owner)
            budget["active_reservations"]["r"]={"metadata":{"experiment":root.name}}
            with self.assertRaises(ValueError):
                score.require_terminal_state(root,budget,lookup=lambda pid:None)

    def test_final_audit_gate_checks_scope_and_zero_reserved(self):
        source="a"*64
        audit=dict(passed=True,accuracy_computed=False,phase="formal",keys=1440,cases=120,
                   own_reserved=0,all_owned_services_stopped=True,run_lock_sha256=source)
        score.require_audit(audit,source)
        for name,value in (("keys",1439),("cases",119),("own_reserved",1),("passed",False),("all_owned_services_stopped",False),("accuracy_computed",True)):
            bad=dict(audit);bad[name]=value
            with self.assertRaises(ValueError):
                score.require_audit(bad,source)

    def test_timing_breakdown_uses_receipts_and_three_hops(self):
        _,rows=fixtures()
        timing=score.timing_components(rows[0])
        self.assertEqual(timing["native_model_generation_seconds"],5)
        self.assertEqual(timing["required_probe_seconds"],4)
        self.assertEqual(timing["geometry_seconds"],6)
        self.assertEqual(timing["selector_and_gather_seconds"],9)
        self.assertEqual(timing["common_state_receiver_probe_seconds"],0)

    def test_humaneval_preflight_records_only_safe_summary(self):
        def fake_executor(code,test,timeout):
            self.assertEqual(timeout,10)
            self.assertIn("def check(",test)
            return {"passed":"intentional evaluator negative fixture" not in code}
        result=run_preflight(DATA,fake_executor)
        self.assertTrue(result["passed"])
        self.assertEqual(result["canonical_passed"],20)
        self.assertEqual(result["wrong_script_rejected"],20)
        serialized=json.dumps(result)
        self.assertNotIn("canonical_solution",serialized)
        self.assertNotIn("def check(",serialized)
        self.assertNotIn("intentional evaluator negative fixture",serialized)

    def test_declared_reference_failure_uses_independent_positive(self):
        tests=json.loads((DATA/"dependencies/humanevalplus_tests.json").read_text(encoding="utf-8"))
        canonical32=tests["HumanEval/32"]["prompt"]+tests["HumanEval/32"]["canonical_solution"]
        def fake_executor(code,test,timeout):
            if "intentional evaluator negative fixture" in code:return {"passed":False,"error_type":"AssertionError"}
            if code==canonical32:return {"passed":False,"error_type":"AssertionError"}
            return {"passed":True}
        result=run_preflight(DATA,fake_executor)
        self.assertTrue(result["passed"])
        self.assertEqual(result["canonical_passed"],19)
        self.assertEqual(result["positive_gate_passed"],20)
        self.assertEqual(result["known_canonical_numeric_failures"],["HumanEval/32"])
        self.assertEqual(result["independent_witness_passed"],1)

    def test_overlay_changes_one_oracle_call_preserves_all_inputs(self):
        from evaluator_overlay import overlay_test,inputs_from_test
        tests=json.loads((DATA/"dependencies/humanevalplus_tests.json").read_text(encoding="utf-8"))
        raw=tests["HumanEval/32"]["rendered_test"]
        effective,proof=overlay_test(raw)
        self.assertEqual(proof["patch_count"],1)
        self.assertEqual(proof["input_count"],886)
        self.assertFalse(proof["absolute_value_added"])
        self.assertEqual(inputs_from_test(raw),inputs_from_test(effective))
        other="def check(candidate): assert candidate(1)==2"
        self.assertEqual(overlay_test(other)[0],other)

    def test_timing_summary_marks_ttft_unknown(self):
        from score import numeric_summary
        summary=numeric_summary([1,2,3])
        self.assertEqual(summary["count"],3)
        self.assertEqual(summary["total_seconds"],6)
        with self.assertRaises(ValueError):
            numeric_summary([float("nan")])
        with self.assertRaises(ValueError):
            numeric_summary([-1])


if __name__=="__main__":
    unittest.main()
