"""CPU-only scheduling, ownership and existing-forward hook checks.

All queues/processes are synthetic. No SSH, GPU allocation, ledger or old-run
files are read or written by these tests.
"""
from __future__ import annotations

import contextlib
import copy
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import common
import manager
from common import choose
from model_ops import capture_prefill
from resource_policy import admit, cpu_set


METHODS = ["Full", "Gen", "H", "L", "H-OBF", "L-OBF", "G", "T", "R", "R-shuffle", "Pseudo-R", "Random-matched"]


def queue_fixture():
    return {"engineering_indices": [0, 20, 40, 60, 80, 100],
            "tasks": [dict(index=i, batch=i//30, state="pending", methods=list(METHODS),
                           committed_methods=[]) for i in range(120)]}


class FakeFile:
    def __init__(self, data):
        self.data = data

    def read_bytes(self):
        if isinstance(self.data, BaseException):
            raise self.data
        return self.data


class FakeProc:
    def __init__(self, pid, command, environment):
        self.name = str(pid)
        self.files = {"cmdline": command, "environ": environment}

    def __truediv__(self, name):
        return FakeFile(self.files[name])


class FakeProcRoot:
    def __init__(self, processes):
        self.processes = processes

    def iterdir(self):
        return iter(self.processes)


def path_factory(processes):
    def factory(path, *parts):
        if str(path) == "/proc" and not parts:
            return FakeProcRoot(processes)
        return Path(path, *parts)
    return factory


class SchedulingTests(unittest.TestCase):
    def test_120_questions_four_30_question_batches(self):
        q = queue_fixture()
        self.assertEqual(len(q["tasks"]), 120)
        for batch in range(4):
            self.assertEqual([t["index"] for t in q["tasks"] if t["batch"] == batch], list(range(batch*30, (batch+1)*30)))
        self.assertEqual(len(q["tasks"])*len(METHODS), 1440)

    def test_engineering_claims_six_indices_across_all_batches(self):
        q = queue_fixture()
        claimed = []
        while (task := choose(q, "engineering")) is not None:
            claimed.append(task["index"])
            # Simulate six concurrent engineering claimants; an earlier running
            # engineering case must not block the other selected groups.
            task["state"] = "running"
        self.assertEqual(claimed, q["engineering_indices"])
        self.assertEqual({q["tasks"][i]["batch"] for i in claimed}, {0, 1, 2, 3})
        self.assertTrue(all(t["state"] == "pending" for t in q["tasks"] if t["index"] not in claimed))

    def test_completed_engineering_answers_are_reused_in_formal(self):
        q = queue_fixture()
        for i in q["engineering_indices"]:
            q["tasks"][i].update(state="complete", methods=[], committed_methods=list(METHODS))
        seen = []
        while (task := choose(q, "formal")) is not None:
            seen.append(task["index"])
            task.update(state="complete", committed_methods=list(METHODS), methods=[])
        self.assertEqual(len(seen), 114)
        self.assertFalse(set(seen) & set(q["engineering_indices"]))
        self.assertEqual(sum(len(t["committed_methods"]) for t in q["tasks"]), 1440)

    def test_running_batch_barrier_prevents_future_batch_claims(self):
        q = queue_fixture()
        for task in q["tasks"][:30]:
            task.update(state="complete", methods=[], committed_methods=list(METHODS))
        q["tasks"][7].update(state="running", methods=["R"], committed_methods=METHODS[:8])
        self.assertIsNone(choose(q, "formal"))
        q["tasks"][7].update(state="complete", methods=[])
        self.assertEqual(choose(q, "formal")["index"], 30)
        q["tasks"][31]["state"] = "running"
        self.assertEqual(choose(q, "formal")["index"], 30)

    def test_claim_records_current_identity_without_other_queue_changes(self):
        q = queue_fixture()
        original = copy.deepcopy(q)
        expected = {"pid": 9001, "uid": 1000, "start_ticks": "123456", "argv": ["python", "worker.py"],
                    "CUDA_VISIBLE_DEVICES": "GPU-owned"}
        saved = []
        with mock.patch.object(common, "locked", side_effect=contextlib.nullcontext), \
             mock.patch.object(common, "load", return_value=q), \
             mock.patch.object(common, "save", side_effect=lambda path, data: saved.append(copy.deepcopy(data))), \
             mock.patch.object(common, "identity", return_value=expected) as identity, \
             mock.patch.object(common.os, "getpid", return_value=9001), \
             mock.patch.object(common.time, "time", return_value=123.0):
            task = common.claim("engineering", Path("synthetic_attempt"))
        identity.assert_called_once_with(9001)
        self.assertEqual(task["owner"], expected)
        self.assertEqual(task["state"], "running")
        self.assertEqual(task["index"], 0)
        self.assertEqual(task["attempt"], "synthetic_attempt")
        self.assertEqual(q["tasks"][1:], original["tasks"][1:])
        self.assertEqual(len(saved), 1)

    def test_commit_rejects_foreign_owner_before_append(self):
        q = queue_fixture()
        q["tasks"][0].update(state="running", owner={"pid": 1777})
        row = dict(dataset="fixture", question_id="synthetic", latent_steps=10, method="R")
        with mock.patch.object(common, "locked", side_effect=contextlib.nullcontext), \
             mock.patch.object(common, "load", return_value=q), \
             mock.patch.object(common.os, "getpid", return_value=9001), \
             mock.patch.object(common, "append") as append:
            with self.assertRaises(AssertionError):
                common.commit(q["tasks"][0], row, Path("synthetic_attempt"))
        append.assert_not_called()


class ResourceAndOwnershipTests(unittest.TestCase):
    def base(self, **overrides):
        result = dict(live_total=0, live_gpu=0, all_ready=True, blocked=False, util=0,
                      free=81920, total=81920, pending=True)
        result.update(overrides)
        return result

    def test_total_six_and_per_card_four_limits(self):
        self.assertFalse(admit(**self.base(live_total=6, live_gpu=2)))
        self.assertFalse(admit(**self.base(live_total=4, live_gpu=4)))
        self.assertTrue(admit(**self.base(live_total=5, live_gpu=3, free=21*1024)))
        self.assertFalse(admit(**self.base(live_total=5, live_gpu=3, free=21*1024-1)))
        self.assertFalse(admit(**self.base(all_ready=False)))
        self.assertFalse(admit(**self.base(blocked=True)))
        self.assertFalse(admit(**self.base(pending=False)))

    def test_fresh_card_load_utilization_and_free_thresholds(self):
        self.assertTrue(admit(**self.base(util=30, free=57344)))
        self.assertFalse(admit(**self.base(util=31)))
        self.assertFalse(admit(**self.base(free=57343)))
        self.assertTrue(admit(**self.base(live_total=1, live_gpu=1, util=100, free=22*1024)))
        self.assertEqual(cpu_set("1-3,8,10-11,3"), [1, 2, 3, 8, 10, 11])

    def test_own_pipe_excludes_foreign_uid_even_same_pipe(self):
        processes = [FakeProc(10, b"python\0worker.py\0", b"CUDA_MPS_PIPE_DIRECTORY=/tmp/owned\0"),
                     FakeProc(11, b"python\0worker.py\0", b"CUDA_MPS_PIPE_DIRECTORY=/tmp/owned\0"),
                     FakeProc(12, b"python\0worker.py\0", b"CUDA_MPS_PIPE_DIRECTORY=/tmp/other\0")]
        identities = {10: dict(pid=10, uid=1000), 11: dict(pid=11, uid=2000), 12: dict(pid=12, uid=1000)}
        with mock.patch.object(manager, "Path", side_effect=path_factory(processes)), \
             mock.patch.object(manager, "identity", side_effect=identities.get), \
             mock.patch.object(manager.os, "getuid", return_value=1000, create=True):
            result = manager.own_pipe("/tmp/owned")
        self.assertEqual(result, [identities[10]])

    def test_foreign_mps_not_mixed_with_owned_service_or_other_gpu(self):
        processes = [FakeProc(20, b"/usr/bin/nvidia-cuda-mps-server\0", b"CUDA_MPS_PIPE_DIRECTORY=/tmp/owned\0CUDA_VISIBLE_DEVICES=GPU-ours\0"),
                     FakeProc(21, b"/usr/bin/nvidia-cuda-mps-server\0", b"CUDA_MPS_PIPE_DIRECTORY=/tmp/foreign\0CUDA_VISIBLE_DEVICES=GPU-ours\0"),
                     FakeProc(22, b"/usr/bin/nvidia-cuda-mps-control\0", b"CUDA_MPS_PIPE_DIRECTORY=/tmp/foreign2\0CUDA_VISIBLE_DEVICES=2\0"),
                     FakeProc(23, b"/usr/bin/nvidia-cuda-mps-control\0", b"CUDA_MPS_PIPE_DIRECTORY=/tmp/unknown\0"),
                     FakeProc(24, b"python\0worker.py\0", b"CUDA_VISIBLE_DEVICES=GPU-ours\0")]
        with mock.patch.object(manager, "Path", side_effect=path_factory(processes)):
            conflicts = manager.foreign_mps({"index": 1, "uuid": "GPU-ours"}, {"/tmp/owned"})
        self.assertEqual(conflicts, [21, 23])


class ExistingPrefillTimingTests(unittest.TestCase):
    def test_hooks_capture_existing_first_forward_without_extra_forward_or_rng(self):
        class FakeCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.Identity()

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeCore()
                self.calls = 0

            def forward(self, hidden):
                self.calls += 1
                normalized = self.model.norm(hidden)
                return types.SimpleNamespace(logits=normalized * 2)

        model = FakeModel()
        source = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        rng_before = torch.random.get_rng_state().clone()
        with mock.patch("torch.cuda.synchronize") as synchronize:
            with capture_prefill(model) as features:
                first = model(source)
                model(source + 100)
        self.assertEqual(model.calls, 2)
        self.assertTrue(torch.equal(features["hidden"], source[:, -1]))
        self.assertTrue(torch.equal(features["logits"], first.logits[:, -1]))
        self.assertGreaterEqual(features["prefill_seconds"], 0)
        self.assertEqual(synchronize.call_count, 2)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model._forward_hooks), 0)
        self.assertEqual(len(model.model.norm._forward_hooks), 0)

    def test_exception_removes_all_hooks(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.norm = torch.nn.Identity()

        model = FakeModel()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with capture_prefill(model):
                raise RuntimeError("fixture")
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model._forward_hooks), 0)
        self.assertEqual(len(model.model.norm._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
