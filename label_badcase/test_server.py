import json
import tempfile
import unittest
import copy
from pathlib import Path

from server import Store, build_dataset, ROOT, handler


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        demo = json.loads((ROOT / "demo_data.json").read_text())
        self.records = build_dataset(demo["badcases"], demo["analysis"], demo["cases"])
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "labels.json"
        self.store = Store(self.records, self.output, {"demo": True})

    def save(self, key, annotation):
        return self.store.mutate({"record_revision": self.store.state["annotations"].get(key, {}).get("revision", 0), "action": "annotation", "key": key, "annotation": annotation})

    def test_completion_validation_and_resume(self):
        a = {"completed": True, "reviews": {"DEMO_RESET": {"verdict": "yes", "reason": "密码重置适用", "owner": "张三"}}}
        self.save("12", a)
        self.assertEqual(self.store.summary()["alternative"], 1)
        self.assertEqual(self.store.summary()["completed"], 1)
        restored = Store(self.records, self.output, {"demo": True})
        self.assertEqual(restored.state["annotations"]["12"]["outcome"], "alternative_found")
        a["reviews"]["DEMO_RESET"]["reason"] = ""
        a["reviews"]["DEMO_RESET"]["verdict"] = "no"
        with self.assertRaises(ValueError):
            self.save("12", a)
        self.assertEqual(self.store.summary()["completed"], 1)

    def test_accept_model_reason_when_yes_and_empty(self):
        self.save("12", {"completed": True, "reviews": {"DEMO_RESET": {
            "verdict": "yes", "reason": " ", "owner": "张三"}}})
        review = self.store.state["annotations"]["12"]["reviews"]["DEMO_RESET"]
        self.assertEqual(review["reason_source"], "model")
        self.assertIn("密码", review["effective_reason"])
        self.assertEqual(review["reason"], " ")
        with self.assertRaises(ValueError):
            self.save("27", {"completed": True, "reviews": {"DEMO_RESET": {
                "verdict": "yes", "reason": "", "owner": "张三"}}})

    def test_uncertain_requires_owner_but_not_reason_or_classification(self):
        self.save("12", {"completed": True, "reviews": {"DEMO_RESET": {
            "verdict": "uncertain", "reason": "", "owner": "张三"}}})
        self.assertEqual(self.store.state["annotations"]["12"]["outcome"], "uncertain")
        self.assertEqual(self.store.summary()["completed"], 1)
        self.assertEqual(self.store.summary()["uncertain"], 1)
        self.assertEqual(self.store.summary()["genuine"], 0)
        self.assertEqual(self.store.summary()["alternative"], 0)
        self.assertEqual(sum(self.store.summary()["error_counts"].values()), 0)
        with self.assertRaises(ValueError):
            self.save("12", {"completed": True, "reviews": {"DEMO_RESET": {
                "verdict": "uncertain", "reason": "", "owner": ""}}})

    def test_drafts_not_counted_as_genuine_and_type_migration(self):
        self.assertEqual(self.store.state["error_types"], ["其他"])
        self.store.mutate({"types_revision": self.store.state["types_revision"], "action": "add_type", "name": "检索排序问题"})
        self.save("20", {"error_type": "检索排序问题", "error_detail": "排序偏差", "owner": "李四"})
        self.assertEqual(self.store.summary()["genuine"], 0)
        self.save("20", {"completed": True, "error_type": "检索排序问题", "error_detail": "排序偏差", "owner": "李四"})
        self.assertEqual(self.store.summary()["error_counts"]["检索排序问题"], 1)
        record_revision = self.store.state["annotations"]["20"]["revision"]
        self.store.mutate({"types_revision": self.store.state["types_revision"], "action": "delete_type", "name": "检索排序问题"})
        a = self.store.state["annotations"]["20"]
        self.assertEqual(a["error_type"], "其他")
        self.assertIn("原错误类型：检索排序问题", a["error_detail"])
        self.assertEqual(self.store.summary()["error_counts"]["其他"], 1)
        self.assertEqual(a["revision"], record_revision + 1)
        with self.assertRaises(RuntimeError):
            self.store.mutate({"record_revision": record_revision, "action": "annotation", "key": "20", "annotation": {}})

    def test_concurrent_revision_rejected_and_gt_not_allowed(self):
        with self.assertRaises(RuntimeError):
            self.store.mutate({"types_revision": -1, "action": "add_type", "name": "新类型"})
        with self.assertRaises(ValueError):
            self.save("12", {"reviews": {"DEMO_GT_RESET": {"verdict": "yes"}}})

    def test_input_change_rejected(self):
        self.save("20", {})
        self.records[0]["query"] = "变化的query"
        with self.assertRaises(ValueError):
            Store(self.records, self.output, {})

    def test_independent_versions_and_same_record_conflict(self):
        self.store.mutate({"record_revision": 0, "action": "annotation", "key": "12", "annotation": {}})
        self.store.mutate({"record_revision": 0, "action": "annotation", "key": "20", "annotation": {}})
        self.assertEqual(self.store.state["annotations"]["20"]["revision"], 1)
        with self.assertRaises(RuntimeError):
            self.store.mutate({"record_revision": 0, "action": "annotation", "key": "12", "annotation": {"owner": "不能覆盖"}})
        self.assertEqual(self.store.state["annotations"]["12"]["owner"], "")

    def test_legacy_file_migration(self):
        self.save("20", {})
        legacy = json.loads(self.output.read_text())
        del legacy["annotations"]["20"]["revision"]
        del legacy["types_revision"]
        self.output.write_text(json.dumps(legacy))
        restored = Store(self.records, self.output, {})
        self.assertEqual(restored.state["annotations"]["20"]["revision"], 0)
        self.assertEqual(restored.state["types_revision"], 0)

    def test_owner_addition_persistence_and_legacy_names(self):
        self.store.mutate({"action": "add_owner", "name": " 张三 "})
        self.store.mutate({"action": "add_owner", "name": "张三"})
        self.assertEqual(self.store.state["owners"], ["张三"])
        self.save("20", {"owner": "李四"})
        self.save("12", {"reviews": {"DEMO_RESET": {"owner": "王五"}}})
        saved = json.loads(self.output.read_text())
        del saved["owners"]
        self.output.write_text(json.dumps(saved))
        restored = Store(self.records, self.output, {})
        self.assertEqual(set(restored.state["owners"]), {"李四", "王五"})
        with self.assertRaises(ValueError):
            self.store.mutate({"action": "add_owner", "name": " "})

    def test_try_retrieval_gt_matching_order_and_no_annotation_write(self):
        before = copy.deepcopy(self.store.state)
        gt = self.store.by_key["12"]["gt"]["case_id"]
        queries = []
        def retrieve(query):
            queries.append(query)
            return {"retrieval_result": {
                "top12": {"case_id": gt, "case_title": "GT标题", "content": "正文", "score": 0.8},
                "top2": {"case_id": "OTHER", "case_title": "GT标题"},
                "metadata": {},
            }}
        self.store.retriever = retrieve
        result = self.store.try_retrieval({"key": "12", "query": "  测试query  "})
        self.assertEqual(queries, ["测试query"])
        self.assertTrue(result["gt_hit"])
        self.assertEqual(result["matched_rank"], 12)
        self.assertEqual([c["rank"] for c in result["cases"]], [2, 12])
        self.assertFalse(result["cases"][0]["is_gt"])
        self.assertEqual(result["cases"][1]["text"], "正文")
        self.assertEqual(before, self.store.state)
        self.assertFalse(self.output.exists())

    def test_try_retrieval_empty_failure_and_validation(self):
        self.store.retriever = lambda query: {"retrieval_result": {}}
        result = self.store.try_retrieval({"key": "12", "query": "没有结果"})
        self.assertFalse(result["gt_hit"])
        self.assertIsNone(result["matched_rank"])
        self.assertEqual(result["cases"], [])
        for body in ({"key": "12", "query": " "}, {"key": "missing", "query": "q"}, [],
                     {"key": "12", "query": "q" * 10001}):
            with self.assertRaises(ValueError):
                self.store.try_retrieval(body)
        self.store.retriever = lambda query: {}
        with self.assertRaisesRegex(RuntimeError, "retrieval_result"):
            self.store.try_retrieval({"key": "12", "query": "q"})
        def fail(query):
            raise TimeoutError("服务超时")
        self.store.retriever = fail
        with self.assertRaisesRegex(RuntimeError, "服务超时"):
            self.store.try_retrieval({"key": "12", "query": "q"})


if __name__ == "__main__":
    unittest.main()
