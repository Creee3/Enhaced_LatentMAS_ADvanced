"""Synthetic CPU checks for final diagnostics aggregation; no model imports."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import summarize_diagnostics as sd


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def geometry_arrays(hop=2, layers=2, heads=2, offset=0):
    real = np.zeros((layers,2,heads,hop,2,3))
    for layer in range(layers):
        for kv in range(2):
            for head in range(heads):
                for source in range(hop):
                    for kind in range(2):
                        real[layer,kv,head,source,kind] = (2+layer+head+source+kv+kind+offset,3+layer+head+source+kv+kind+offset,2)
    overlap = np.full((layers,2,heads,2,hop,hop), .2)
    for source in range(hop):
        overlap[...,source,source] = 1
    counts = np.full((layers,heads,hop,2),8,dtype=np.int32)
    present = np.ones((layers,heads,2),bool)
    return dict(real_metrics=real,pseudo_metrics=real+np.array([1,1,0]),
        real_sample_count=counts,pseudo_sample_count=counts.copy(),
        real_all_sources_present=present,pseudo_all_sources_present=present.copy(),
        real_subspace_overlap=overlap,pseudo_subspace_overlap=np.where(overlap==1,1,.5),
        real_subspace_overlap_rank=np.full(overlap.shape,2,np.int16),pseudo_subspace_overlap_rank=np.full(overlap.shape,2,np.int16),
        pseudo_degenerate_by_head=np.full((layers,heads),hop==1,bool),
        pseudo_changed_fraction_by_head=np.full((layers,heads),0 if hop==1 else .5))


def readout_arrays(hop=2, layers=2, heads=2, factor=1):
    cross = np.zeros((layers,heads,hop+1,hop+1))
    cross[...,0,0] = .25*factor
    for source in range(1,hop+1):
        cross[...,source,source] = factor
    if hop>=2:
        cross[...,1,2] = cross[...,2,1] = -.5*factor
    energy = np.diagonal(cross,axis1=-2,axis2=-1).copy()
    total = cross.sum((-2,-1))
    denom = np.sqrt(energy[..., :,None]*energy[...,None,:])
    cosine = np.zeros_like(cross)
    np.divide(cross,denom,out=cosine,where=denom>0)
    esum = energy.sum(-1)
    ratio = np.zeros_like(total)
    np.divide(total,esum,out=ratio,where=esum>0)
    return dict(source_error_mse=energy,total_error_mse=total,source_error_cross_matrix=cross,
        source_error_cosine_matrix=cosine,cancellation=np.where(esum>0,1-ratio,0),zero_error=esum==0,
        source_error_mean_vector=np.zeros((layers,heads,hop+1,128)),
        total_error_mean_vector=np.zeros((layers,heads,128)))


def receiver_arrays(value=2):
    return dict(full_to_selected_kl=np.array([value],float),hidden_relative_l2=np.array([value/10]),
        hidden_cosine=np.array([.9]),full_hidden_zero=np.array([False]))


def store_npz(path, arrays, metadata):
    payload=dict(arrays,metadata_json=np.asarray(json.dumps(metadata)))
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**payload)
    return dict(file=path.name,sha256=sd.file_sha(path),bytes=path.stat().st_size,
        arrays=sorted(payload),readback_verified=True,raw_kv_retained=False)


def geo_metadata(method,hop,source):
    return dict(method=method,hop=hop,source_cache_sha=source,geometry_version=1,feature_subspace_rank=4,
        token_class_names=["prompt","latent"],metric_names=["participation_ratio","entropy_effective_rank","numerical_rank"],source_names=list(sd.ROLES[:hop]))


def final_fixture(root):
    dataset = {"AIME24":"aime2024","GPQA-Diamond":"gpqa","GPQA-nonDiamond":"gpqa","HumanEval+":"humanevalplus","MedQA":"medqa","MMLU-Pro":"mmlu_pro"}
    items=[]
    for group in sd.GROUPS:
        for n in range(20):
            i=len(items)
            items.append(dict(index=i,global_index=i,dataset=dataset[group],dataset_group=group,
                question_id=f"{group}-{n}",category="test",is_diamond=group=="GPQA-Diamond",question="PRIVATE QUESTION NEVER REPORT"))
    write_json(root/"questions.json",items)
    write_json(root/"data_manifest.json",dict(questions_sha256=sd.file_sha(root/"questions.json")))
    write_json(root/"plan.json",dict(target_keys=1440,methods=list(sd.METHODS)))
    lock={"files":{name:dict(bytes=(root/name).stat().st_size,sha256=sd.file_sha(root/name)) for name in ("questions.json","data_manifest.json","plan.json")}}
    write_json(root/"run_lock.json",lock)
    source=sd.file_sha(root/"run_lock.json")
    write_json(root/"formal_audit_verified.json",dict(passed=True,phase="formal",keys=1440,cases=120,own_reserved=0,all_owned_services_stopped=True,accuracy_computed=False,run_lock_sha256=source))
    write_json(root/"formal_archive_verified.json",dict(passed=True,tag="formal",remote_stream_sha=True,gzip_crc=True,exact_members=True,per_member_sha=True,safe_full_extract_readback_sha=True,archive_sha256="c"*64))
    write_json(root/"queue.json",dict(tasks=[dict(state="complete",methods=[]) for _ in items]))
    results=[]
    for item in items:
        for method in sd.METHODS:
            results.append(dict(**item,method=method,latent_steps=10,seed=42,status="ok",source_lock_sha=source,
                diagnostic_files=[],common_state_diagnostic_files=[],common_state_diagnostic_request_ids=[],request_ids=[],transfers=[],
                shared_planner_kv_sha="d"*64,fork_rng_sha="e"*64,shared_request_ids=[],answer_text="PRIVATE ANSWER NEVER REPORT",generated_ids=[999]))
    path=root/"attempts/a/results.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in results),encoding="utf-8")
    return items,results,path


class SummaryTests(unittest.TestCase):
    def test_geometry_question_reduction_masks_pseudo_and_offdiagonal(self):
        arrays=geometry_arrays()
        # One head has a degenerate pseudo partition; real data remains usable.
        arrays["pseudo_degenerate_by_head"][0,0]=True
        arrays["real_subspace_overlap_rank"][1,:,1,:,0,1]=0
        arrays["real_subspace_overlap_rank"][1,:,1,:,1,0]=0
        out=sd.reduce_geometry(arrays,2)
        self.assertEqual(out["geometry.K.prompt.real.participation_ratio"],3.5)
        self.assertEqual(out["geometry.V.latent.real.participation_ratio"],5.5)
        self.assertEqual(out["geometry.K.prompt.real_minus_pseudo.participation_ratio"],-1)
        self.assertAlmostEqual(out["geometry.K.prompt.real.top4_subspace_overlap_offdiag"],.2)
        self.assertAlmostEqual(out["geometry.K.prompt.real_minus_pseudo.top4_subspace_overlap_offdiag"],-.3)
        self.assertEqual(out["geometry.pseudo_degenerate_fraction"],.25)
        first=sd.reduce_geometry(geometry_arrays(hop=1),1)
        self.assertIsNone(first["geometry.K.prompt.real_minus_pseudo.participation_ratio"])
        self.assertIsNone(first["geometry.K.prompt.real.top4_subspace_overlap_offdiag"])
        self.assertGreater(first["geometry.K.prompt.real.participation_ratio"],0)

    def test_geometry_missing_and_rank_zero_not_counted_as_zero(self):
        arrays=geometry_arrays()
        arrays["real_metrics"][0,:,0,:,0,:]=0
        arrays["real_sample_count"][0,0,:,0]=0
        arrays["real_all_sources_present"][0,0,0]=False
        out=sd.reduce_geometry(arrays,2)
        self.assertEqual(out["geometry.K.prompt.real_valid_component_fraction"],.75)
        self.assertAlmostEqual(out["geometry.K.prompt.real.participation_ratio"],(3.5+3.5+4.5)/3)

    def test_true_source_error_cosine_and_cancellation_exclude_sink(self):
        out=sd.reduce_readout(readout_arrays(),2)
        self.assertEqual(out["readout.true_source_error_mse"],1)
        self.assertEqual(out["readout.total_error_mse"],1.25)
        self.assertEqual(out["readout.cancellation_true_sources"],.5)
        self.assertAlmostEqual(out["readout.cancellation_including_sink_self"],1-1.25/2.25)
        self.assertEqual(out["readout.true_source_error_cosine_offdiag"],-.5)
        zero=sd.reduce_readout(readout_arrays(factor=0),2)
        self.assertEqual(zero["readout.total_error_mse"],0)
        self.assertIsNone(zero["readout.cancellation_true_sources"])
        self.assertIsNone(zero["readout.true_source_error_cosine_offdiag"])
        bad=readout_arrays();bad["total_error_mse"]+=1
        with self.assertRaisesRegex(ValueError,"cross-term"):
            sd.reduce_readout(bad,2)

    def test_readout_never_loads_vector_members(self):
        class Lazy(dict):
            def __getitem__(self,name):
                if name.endswith("_vector"):
                    raise AssertionError("raw derived vector loaded")
                return super().__getitem__(name)
        self.assertGreater(sd.reduce_readout(Lazy(readout_arrays()),2)["readout.total_error_mse"],0)

    def test_receiver_zero_reference_norm_is_missing(self):
        arrays=receiver_arrays();arrays["full_hidden_zero"][:]=True
        out=sd.reduce_receiver(arrays)
        self.assertEqual(out["receiver.full_to_selected_kl"],2)
        self.assertIsNone(out["receiver.hidden_relative_l2"])
        self.assertIsNone(out["receiver.hidden_cosine"])

    def test_bootstrap_resamples_questions_and_records_missing(self):
        records=[dict(x=float(i),missing=None,one=7 if i==0 else None) for i in range(20)]
        a=sd.bootstrap_questions(records,iterations=1000,seed=42)
        self.assertEqual(a,sd.bootstrap_questions(records,iterations=1000,seed=42))
        self.assertEqual(a["x"]["n_questions"],20)
        self.assertEqual(a["x"]["mean"],9.5)
        self.assertLess(a["x"]["ci95_lower"],9.5)
        self.assertGreater(a["x"]["ci95_upper"],9.5)
        self.assertEqual(a["missing"]["missing_questions"],20)
        self.assertIsNone(a["missing"]["mean"])
        self.assertIsNone(a["one"]["ci95_lower"])

    def test_npz_hash_metadata_and_path_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/"H_hop2_readout.npz"
            meta=dict(method="H",hop=2,analysis="own_trajectory_same_state",source_cache_sha="a",position_sha="b")
            receipt=store_npz(path,readout_arrays(),meta)
            output,_=sd.read_diagnostic(root,receipt,method="H",hop=2,kind="readout",source_sha="a",position_sha="b")
            self.assertEqual(output["readout.total_error_mse"],1.25)
            with self.assertRaisesRegex(ValueError,"method/hop"):
                sd.read_diagnostic(root,receipt,method="R",hop=2,kind="readout")
            with self.assertRaisesRegex(ValueError,"source-cache"):
                sd.read_diagnostic(root,receipt,method="H",hop=2,kind="readout",source_sha="wrong")
            escape=dict(receipt,file="../escape.npz")
            with self.assertRaisesRegex(ValueError,"confined"):
                sd.read_diagnostic(root,escape,method="H",hop=2,kind="readout")
            with path.open("ab") as f:f.write(b"changed")
            with self.assertRaisesRegex(ValueError,"SHA/size"):
                sd.read_diagnostic(root,receipt,method="H",hop=2,kind="readout")

    def test_final_gates_and_exact_coverage_before_any_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);items,results,path=final_fixture(root)
            _,minimal,descriptors,_=sd.require_final_inputs(root)
            self.assertEqual(len(minimal),120);self.assertEqual(len(descriptors),1440)
            self.assertNotIn("question",minimal[0])
            self.assertNotIn("answer_text",next(iter(descriptors.values())))
            self.assertNotIn("generated_ids",next(iter(descriptors.values())))
            path.write_text("".join(json.dumps(r)+"\n" for r in results[:-1]),encoding="utf-8")
            with mock.patch("summarize_diagnostics.np.load",side_effect=AssertionError("NPZ touched before scope gate")):
                with self.assertRaisesRegex(ValueError,"1440 exact"):
                    sd.build_summary(root)
            write_json(root/"formal_archive_verified.json",dict(passed=False))
            with self.assertRaisesRegex(ValueError,"raw formal archive"):
                sd.require_final_inputs(root)

    def test_one_question_all12_scopes_hops_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);item=dict(index=0,dataset="gpqa",question_id="private")
            descriptors={}
            common=[];common_ids=[f"common-{m}-{hop}" for hop in (2,3) for m in sd.METHODS if m!="Full"]
            for method in sd.METHODS:
                files=[];transfers=[]
                req=["planner","firstprobe",method+"critic",method+"probe2",method+"refiner",method+"probe3",method+"judger"]
                for hop in (1,2,3):
                    source="shared" if hop==1 else "Full"+str(hop) if method=="Full" else method+str(hop)
                    position="pos"+method+str(hop)
                    transfers.append(dict(position_sha=position))
                    files.append(store_npz(root/f"{method}_hop{hop}_geometry.npz",geometry_arrays(hop,layers=1,heads=1,offset=0 if method=="Full" else 10),geo_metadata(method,hop,source)))
                    files.append(store_npz(root/f"{method}_hop{hop}_readout.npz",readout_arrays(hop,layers=1,heads=1),dict(method=method,hop=hop,analysis="own_trajectory_same_state",source_cache_sha=source,position_sha=position)))
                    files.append(store_npz(root/f"{method}_hop{hop}_receiver.npz",receiver_arrays(2),dict(method=method,hop=hop,analysis="own_trajectory_same_state",first_hop_common_prefix=hop==1,receiver=sd.RECEIVERS[hop-1],decode_position=0,reference_probe_rid=req[{1:1,2:3,3:5}[hop]],native_prefill_rid=req[{1:2,2:4,3:6}[hop]])))
                descriptors[("gpqa","private",10,method)]=dict(_attempt=root,diagnostic_files=files,transfers=transfers,request_ids=req,shared_planner_kv_sha="shared",fork_rng_sha="rng",shared_request_ids=["planner","firstprobe"],common_state_diagnostic_request_ids=common_ids)
            for method in sd.METHODS:
                for hop in (1,2,3):
                    arrays=readout_arrays(hop,layers=1,heads=1)
                    meta=dict(method=method,hop=hop,analysis="same_full_state",source_cache_sha="shared" if hop==1 else "Full"+str(hop))
                    if hop>1 and method!="Full":
                        arrays.update({"receiver_"+k:v for k,v in receiver_arrays(7).items()})
                        meta["receiver_probe_rid"]=f"common-{method}-{hop}"
                    common.append(store_npz(root/f"common_hop{hop}_{method}.npz",arrays,meta))
            for row in descriptors.values():row["common_state_diagnostic_files"]=common
            own,shared,n=sd.reduce_question(item,descriptors)
            self.assertEqual(n,144)
            self.assertEqual(own["H"][2]["receiver.full_to_selected_kl"],2)
            self.assertEqual(shared["H"][2]["receiver.full_to_selected_kl"],7)
            self.assertEqual(shared["H"][1]["receiver.full_to_selected_kl"],2)
            self.assertIsNone(shared["Full"][2]["receiver.full_to_selected_kl"])
            name="geometry.K.prompt.real.participation_ratio"
            self.assertEqual(shared["H"][2][name],own["Full"][2][name])
            self.assertNotEqual(shared["H"][2][name],own["H"][2][name])

    def test_six_groups_are_independent_and_private_output_has_no_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);final_fixture(root)
            def fake_reduce(item,descriptors):
                group=sd.GROUPS.index(item["dataset_group"])
                metrics={"readout.true_source_error_mse":float(group),"readout.total_error_mse":float(group),"readout.cancellation_including_sink_self":None,"readout.true_source_error_cosine_offdiag":None,"receiver.full_to_selected_kl":float(group),"receiver.hidden_relative_l2":0.0}
                scope={m:{h:dict(metrics) for h in (1,2,3)} for m in sd.METHODS}
                return scope,copy.deepcopy(scope),144
            with mock.patch("summarize_diagnostics.reduce_question",side_effect=fake_reduce):
                proof=sd.summarize_run(root,iterations=50,seed=42)
            self.assertEqual(proof["keys"],1440)
            self.assertEqual(proof["verified_diagnostic_files"],17280)
            result=sd.load_json(root/"final_diagnostics_summary.json")
            self.assertEqual(result["datasets"]["MedQA"]["own_trajectory"]["H"]["hops"]["2"]["metrics"]["readout.total_error_mse"]["mean"],4)
            self.assertEqual(result["datasets"]["AIME24"]["own_trajectory"]["H"]["hops"]["2"]["metrics"]["readout.total_error_mse"]["mean"],0)
            self.assertNotIn("pooled",result["datasets"])
            text=(root/"final_diagnostics_summary.json").read_text()+(root/"final_diagnostics_summary.md").read_text()
            for private in ("PRIVATE QUESTION", "PRIVATE ANSWER", "generated_ids", "AIME24-0"):
                self.assertNotIn(private,text)
            self.assertEqual(proof["model_forward_calls"],0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
