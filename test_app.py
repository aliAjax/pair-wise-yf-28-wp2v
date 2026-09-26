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
    def _block_lengths(self, trial_id):
        """返回每个分层每个区组实际包含的分配编号数（按需生成后可见）。"""
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT stratum_id,block_no,COUNT(*) AS n FROM allocations WHERE trial_id=? GROUP BY stratum_id,block_no",
                (trial_id,),
            ).fetchall()
        return [(r["stratum_id"], r["block_no"], r["n"]) for r in rows]

    def test_block_size_revision_requires_other_approver_and_only_affects_unused_blocks(self):
        # 旧长度 4：先入组填满区组 1（4 例）
        for i in range(1, 5):
            self.store.enroll("site1", self.trial["id"], f"S001-{i:03d}", {"risk": "low"})
        revision = self.store.request_block_size_revision(
            "coord", self.trial["id"], 8, "中心实际入组速度高于预期，需调整区组长度平衡组间分配"
        )
        self.assertEqual(revision["status"], "pending")
        self.assertEqual(revision["old_block_size"], 4)
        self.assertEqual(revision["new_block_size"], 8)

        # 待审期间旧方案继续承接入组：区组 2 仍按旧长度 4 生成
        self.store.enroll("site1", self.trial["id"], "S001-005", {"risk": "low"})
        with self.store.connect() as conn:
            current = conn.execute("SELECT block_size FROM trials WHERE id=?", (self.trial["id"],)).fetchone()[0]
        self.assertEqual(current, 4)

        # 监查员能看到待审版本；站点角色不能查看修订
        pending = self.store.list_block_size_revisions("monitor1", self.trial["id"], "pending")
        self.assertEqual([r["id"] for r in pending], [revision["id"]])
        with self.assertRaises(BusinessError) as ctx:
            self.store.list_block_size_revisions("site1", self.trial["id"])
        self.assertEqual(ctx.exception.code, "forbidden")

        # 申请人和审批人相同就拒绝
        with self.assertRaises(BusinessError) as ctx:
            self.store.decide_block_size_revision("coord", revision["id"], True)
        self.assertEqual(ctx.exception.code, "self_review_forbidden")

        # 另一人审批通过
        approved = self.store.decide_block_size_revision("monitor1", revision["id"], True, note="同意")
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["reviewer_id"], "monitor1")
        with self.store.connect() as conn:
            self.assertEqual(conn.execute("SELECT block_size FROM trials WHERE id=?", (self.trial["id"],)).fetchone()[0], 8)

        # 新长度只影响尚未使用的区组：区组 1、2 保持 4，区组 3 起为 8
        for i in range(5, 13):
            self.store.enroll("site1", self.trial["id"], f"S001-{i:03d}", {"risk": "low"})
        lengths = {block_no: n for _, block_no, n in self._block_lengths(self.trial["id"])}
        self.assertEqual(lengths[1], 4)
        self.assertEqual(lengths[2], 4)
        self.assertEqual(lengths[3], 8)

        # 每个区组仍按分组数均衡；既往审计不受影响且可回看
        with self.store.connect() as conn:
            balance = conn.execute(
                "SELECT block_no, arm, COUNT(*) AS n FROM allocations WHERE trial_id=? GROUP BY block_no, arm",
                (self.trial["id"],),
            ).fetchall()
        counts = {(r["block_no"], r["arm"]): r["n"] for r in balance}
        self.assertEqual(set(counts[(3, a)] for a in ("A", "B")), {4})
        summary = self.store.trial_summary("coord", self.trial["id"])
        actions = [a["action"] for a in summary["audit"]]
        self.assertIn("block_revision.request", actions)
        self.assertIn("block_revision.approve", actions)
        self.assertEqual(len(summary["block_size_revisions"]), 1)

    def test_block_size_revision_rejected_leaves_trial_unchanged(self):
        self.store.enroll("site1", self.trial["id"], "S001-001", {"risk": "low"})
        revision = self.store.request_block_size_revision(
            "coord", self.trial["id"], 6, "入组速度与原定区组长度不匹配需要修订"
        )
        result = self.store.decide_block_size_revision("monitor1", revision["id"], False, note="依据不足")
        self.assertEqual(result["status"], "rejected")
        with self.store.connect() as conn:
            self.assertEqual(conn.execute("SELECT block_size FROM trials WHERE id=?", (self.trial["id"],)).fetchone()[0], 4)
        # 拒绝后继续入组，新生成的区组仍是旧长度 4
        for i in range(2, 5):
            self.store.enroll("site1", self.trial["id"], f"S001-{i:03d}", {"risk": "low"})
        self.assertTrue(all(n == 4 for _, _, n in self._block_lengths(self.trial["id"])))
        # 拒绝后可重新提交
        again = self.store.request_block_size_revision("coord", self.trial["id"], 8, "补充材料后再次申请调整长度")
        self.assertEqual(again["status"], "pending")
        self.assertEqual(len(self.store.list_block_size_revisions("coord", self.trial["id"])), 2)

    def test_block_size_revision_guardrails(self):
        # 只有进行中的试验可以提交
        draft = self.store.create_trial("coord", "另一项观察研究", "v1.0", ["A", "B"], ["risk"], 4, "seed-2026-002")
        with self.assertRaises(BusinessError) as ctx:
            self.store.request_block_size_revision("coord", draft["id"], 8, "草稿阶段不应提交区组长度修订")
        self.assertEqual(ctx.exception.code, "trial_not_running")
        # 站点角色不能提交；长度必须是分组数的整数倍
        with self.assertRaises(BusinessError) as ctx:
            self.store.request_block_size_revision("site1", self.trial["id"], 8, "中心人员尝试自行修订区组长度")
        self.assertEqual(ctx.exception.code, "forbidden")
        with self.assertRaises(BusinessError) as ctx:
            self.store.request_block_size_revision("coord", self.trial["id"], 5, "长度不是试验组数量的整数倍")
        self.assertEqual(ctx.exception.code, "invalid_block_size")
        # 同一时间只允许一个待审版本
        self.store.request_block_size_revision("coord", self.trial["id"], 8, "第一次提交区组长度修订申请")
        with self.assertRaises(BusinessError) as ctx:
            self.store.request_block_size_revision("coord", self.trial["id"], 12, "待审版本尚未处理又提交一次")
        self.assertEqual(ctx.exception.code, "revision_pending")
        # 同一修订不能重复审批
        first = self.store.list_block_size_revisions("monitor1", self.trial["id"])[0]
        self.store.decide_block_size_revision("monitor1", first["id"], True)
        with self.assertRaises(BusinessError) as ctx:
            self.store.decide_block_size_revision("monitor2", first["id"], True)
        self.assertEqual(ctx.exception.code, "already_decided")


if __name__ == "__main__":
    unittest.main()
