import tempfile
import unittest
from collections import Counter
from pathlib import Path

from app import BusinessError, RandomizationStore


class RandomizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RandomizationStore(Path(self.tmp.name) / "test.db")
        self.store.seed()
        self.trial = self.store.create_trial(
            "coord", "多中心降压研究", "v1.0", ["A", "B"], ["risk"], 4, "seed-2026-001"
        )
        self.store.start_trial("coord", self.trial["id"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_stratified_block_randomization_and_two_person_unblinding(self):
        participants = [
            self.store.enroll("site1", self.trial["id"], f"S001-{i:03d}", {"risk": "low"})
            for i in range(1, 5)
        ]
        self.assertNotIn("arm", participants[0])
        with self.store.connect() as conn:
            arms = [r["arm"] for r in conn.execute(
                "SELECT a.arm FROM allocations a JOIN participants p ON p.allocation_id=a.id WHERE p.trial_id=? ORDER BY p.id",
                (self.trial["id"],),
            ).fetchall()]
        self.assertEqual(Counter(arms), Counter({"A": 2, "B": 2}))
        request = self.store.request_unblinding("site1", participants[0]["id"], "受试者发生严重不良事件需要紧急处理")
        first = self.store.approve_unblinding("monitor1", request["id"])
        self.assertEqual(first["status"], "pending")
        with self.assertRaises(BusinessError) as ctx:
            self.store.approve_unblinding("monitor1", request["id"])
        self.assertEqual(ctx.exception.code, "distinct_approver_required")
        second = self.store.approve_unblinding("monitor2", request["id"])
        self.assertEqual(second["status"], "approved")
        self.assertIn(second["arm"], {"A", "B"})

    def test_idempotent_enrollment_site_isolation_and_protocol_lock(self):
        first = self.store.enroll("site1", self.trial["id"], "S001-001", {"risk": "high"})
        again = self.store.enroll("site1", self.trial["id"], "S001-001", {"risk": "high"})
        self.assertEqual(first["id"], again["id"])
        self.assertTrue(again["idempotent"])
        with self.store.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM participants").fetchone()[0], 1)
        with self.assertRaises(BusinessError) as ctx:
            self.store.get_participant("site2", first["id"])
        self.assertEqual(ctx.exception.code, "site_isolation")
        with self.assertRaises(BusinessError) as ctx:
            self.store.update_protocol("coord", self.trial["id"], "v2")
        self.assertEqual(ctx.exception.code, "protocol_locked")


class BlockSizeRevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RandomizationStore(Path(self.tmp.name) / "test.db")
        self.store.seed()
        self.trial = self.store.create_trial(
            "coord", "区组修订研究", "v1.0", ["A", "B"], ["risk"], 2, "seed-2026-002"
        )
        self.store.start_trial("coord", self.trial["id"])

    def tearDown(self):
        self.tmp.cleanup()

    def _allocations(self):
        with self.store.connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT sequence,block_no,arm,used_by,used_at FROM allocations WHERE trial_id=? ORDER BY sequence",
                (self.trial["id"],),
            ).fetchall()]

    def test_approved_revision_affects_only_future_blocks(self):
        self.store.enroll("site1", self.trial["id"], "S001-001", {"risk": "low"})
        self.store.enroll("site1", self.trial["id"], "S001-002", {"risk": "low"})
        before = self._allocations()
        self.assertEqual([a["block_no"] for a in before], [1, 1])

        revision = self.store.submit_block_size_revision(
            "coord", self.trial["id"], 4, "中心入组速度高于预期需调整区组"
        )
        self.assertEqual(revision["status"], "pending")
        self.assertEqual(revision["current_block_size"], 2)

        # 待审期间旧方案继续承接入组：第二个区组仍按长度 2 生成
        self.store.enroll("site1", self.trial["id"], "S001-003", {"risk": "low"})
        pending = self._allocations()
        self.assertEqual(len([a for a in pending if a["block_no"] == 2]), 2)

        # 监查员可见待审版本
        items = self.store.list_block_size_revisions("monitor1", self.trial["id"])
        self.assertEqual([(i["status"], i["proposed_block_size"]) for i in items], [("pending", 4)])

        # 申请人与审批人相同则拒绝
        with self.assertRaises(BusinessError) as ctx:
            self.store.decide_block_size_revision("coord", revision["id"], True)
        self.assertEqual(ctx.exception.code, "self_decision_forbidden")

        approved = self.store.decide_block_size_revision("monitor1", revision["id"], True)
        self.assertEqual((approved["status"], approved["decided_by"]), ("approved", "monitor1"))

        self.store.enroll("site1", self.trial["id"], "S001-004", {"risk": "low"})
        self.store.enroll("site1", self.trial["id"], "S001-005", {"risk": "low"})
        after = self._allocations()
        # 批准后新长度只影响尚未使用的区组：第三个区组长度为 4
        self.assertEqual(len([a for a in after if a["block_no"] == 3]), 4)
        # 已发出的编号与盲底保持原样
        self.assertEqual([a for a in after if a["block_no"] == 1], before)
        # 既往审计保持原样，新事件追加在末尾
        summary = self.store.trial_summary("monitor2", self.trial["id"])
        actions = [a["action"] for a in summary["audit"]]
        self.assertEqual(actions[0], "trial.create")
        self.assertLess(
            actions.index("block_size_revision.submit"),
            actions.index("block_size_revision.approve"),
        )
        self.assertEqual(summary["trial"]["block_size"], 4)

    def test_rejection_leaves_protocol_unchanged(self):
        self.store.enroll("site1", self.trial["id"], "S001-001", {"risk": "high"})
        revision = self.store.submit_block_size_revision(
            "coord", self.trial["id"], 4, "中心入组速度与区组长度不匹配"
        )
        rejected = self.store.decide_block_size_revision("monitor1", revision["id"], False)
        self.assertEqual(rejected["status"], "rejected")
        with self.store.connect() as conn:
            size = conn.execute(
                "SELECT block_size FROM trials WHERE id=?", (self.trial["id"],)
            ).fetchone()[0]
        self.assertEqual(size, 2)
        again = self.store.submit_block_size_revision(
            "coord", self.trial["id"], 6, "拒绝后重新评估再次提交修订"
        )
        self.assertEqual(again["status"], "pending")

    def test_revision_validation(self):
        with self.assertRaises(BusinessError) as ctx:
            self.store.submit_block_size_revision("coord", self.trial["id"], 3, "长度必须为分组整数倍")
        self.assertEqual(ctx.exception.code, "invalid_block_size")
        with self.assertRaises(BusinessError) as ctx:
            self.store.submit_block_size_revision("coord", self.trial["id"], 2, "与现行长度相同的申请")
        self.assertEqual(ctx.exception.code, "unchanged_block_size")
        with self.assertRaises(BusinessError) as ctx:
            self.store.submit_block_size_revision("coord", self.trial["id"], 4, "太短")
        self.assertEqual(ctx.exception.code, "reason_required")
        with self.assertRaises(BusinessError) as ctx:
            self.store.submit_block_size_revision("site1", self.trial["id"], 4, "中心用户不能提交修订")
        self.assertEqual(ctx.exception.code, "forbidden")
        revision = self.store.submit_block_size_revision(
            "coord", self.trial["id"], 4, "合理的修订原因说明"
        )
        with self.assertRaises(BusinessError) as ctx:
            self.store.submit_block_size_revision("coord", self.trial["id"], 6, "已有待审修订时再次提交")
        self.assertEqual(ctx.exception.code, "revision_pending")
        self.store.decide_block_size_revision("monitor1", revision["id"], True)
        with self.assertRaises(BusinessError) as ctx:
            self.store.decide_block_size_revision("monitor2", revision["id"], True)
        self.assertEqual(ctx.exception.code, "already_decided")

    def test_revision_requires_running_trial(self):
        draft = self.store.create_trial(
            "coord", "未启动试验", "v1.0", ["A", "B"], ["risk"], 2, "seed-2026-003"
        )
        with self.assertRaises(BusinessError) as ctx:
            self.store.submit_block_size_revision("coord", draft["id"], 4, "试验尚未启动不能修订")
        self.assertEqual(ctx.exception.code, "invalid_status")


if __name__ == "__main__":
    unittest.main()
