import json
import tempfile
import unittest
from pathlib import Path

from server import Store, build_dataset, ROOT


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        demo = json.loads((ROOT / "demo_data.json").read_text())
        self.records = build_dataset(demo["badcases"], demo["analysis"], demo["cases"])
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "labels.json"
        self.store = Store(self.records, self.output, {"demo": True})

    def save(self, key, annotation):
        return self.store.mutate({"revision": self.store.state["revision"], "action": "annotation", "key": key, "annotation": annotation})

    def test_completion_validation_and_resume(self):
        a = {"completed": True, "reviews": {"DEMO_RESET": {"verdict": "yes", "reason": "密码重置适用", "owner": "张三"}}}
        self.save("12", a)
        self.assertEqual(self.store.summary()["alternative"], 1)
        self.assertEqual(self.store.summary()["completed"], 1)
        restored = Store(self.records, self.output, {"demo": True})
        self.assertEqual(restored.state["annotations"]["12"]["outcome"], "alternative_found")
        a["reviews"]["DEMO_RESET"]["reason"] = ""
        with self.assertRaises(ValueError):
            self.save("12", a)
        self.assertEqual(self.store.summary()["completed"], 1)

    def test_drafts_not_counted_as_genuine_and_type_migration(self):
        self.save("20", {"error_type": "检索排序问题", "error_detail": "排序偏差", "owner": "李四"})
        self.assertEqual(self.store.summary()["genuine"], 0)
        self.save("20", {"completed": True, "error_type": "检索排序问题", "error_detail": "排序偏差", "owner": "李四"})
        self.assertEqual(self.store.summary()["error_counts"]["检索排序问题"], 1)
        self.store.mutate({"revision": self.store.state["revision"], "action": "delete_type", "name": "检索排序问题"})
        a = self.store.state["annotations"]["20"]
        self.assertEqual(a["error_type"], "其他")
        self.assertIn("原错误类型：检索排序问题", a["error_detail"])
        self.assertEqual(self.store.summary()["error_counts"]["其他"], 1)

    def test_concurrent_revision_rejected_and_gt_not_allowed(self):
        with self.assertRaises(RuntimeError):
            self.store.mutate({"revision": -1, "action": "add_type", "name": "新类型"})
        with self.assertRaises(ValueError):
            self.save("12", {"reviews": {"DEMO_GT_RESET": {"verdict": "yes"}}})

    def test_input_change_rejected(self):
        self.save("20", {})
        self.records[0]["query"] = "变化的query"
        with self.assertRaises(ValueError):
            Store(self.records, self.output, {})


if __name__ == "__main__":
    unittest.main()
