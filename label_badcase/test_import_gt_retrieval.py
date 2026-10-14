import tempfile
import unittest
from pathlib import Path

from import_gt_retrieval import import_report, main
from server import ROOT, Store, atomic_write, build_dataset, load


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        demo = load(ROOT / "demo_data.json")
        self.records = build_dataset(demo["badcases"], demo["analysis"], demo["cases"])
        sources = {}
        for name in ("badcases", "analysis", "cases"):
            source = self.path / (name + ".json")
            atomic_write(source, demo[name])
            sources[name] = str(source)
        self.store = Store(self.records, self.path / "labels.json", sources)
        self.report = {"artifact_type": "gt_title_retrieval_check", "misses": [
            {"expected_case_id": r["gt"]["case_id"], "outcome": "miss", "matched_rank": None}
            for r in self.records]}

    def test_filter_completion_and_idempotence(self):
        expected = {r["key"] for r in self.records if r["model_status"] == "success"
                    and not any(c["model_same"] is True and not c["is_gt"] for c in r["retrieved"])}
        stats = import_report(self.store, self.report)
        self.assertTrue(expected)
        self.assertEqual(set(stats["imported_keys"]), expected)
        for key in expected:
            a = self.store.state["annotations"][key]
            self.assertEqual(a["error_type"], "检索端问题")
            self.assertEqual(a["error_detail"], "即使用GT案例标题检索也检索不到")
            self.assertEqual(a["owner"], "唐锦康")
            self.assertEqual(a["outcome"], "genuine_badcase")
            self.assertTrue(a["completed"])
        self.assertEqual(import_report(self.store, self.report)["imported"], 0)

    def test_preserve_existing_and_overwrite(self):
        self.store.state["annotations"]["20"] = {"revision": 4, "reviews": {}, "owner": "原负责人"}
        self.assertEqual(import_report(self.store, self.report)["existing_skipped"], 1)
        self.assertEqual(self.store.state["annotations"]["20"]["owner"], "原负责人")
        self.assertIn("20", import_report(self.store, self.report, overwrite=True)["imported_keys"])
        self.assertEqual(self.store.state["annotations"]["20"]["revision"], 5)

    def test_outside_cutoff_is_not_absent_and_reviews_protected(self):
        for row in self.report["misses"]:
            row["matched_rank"] = 12
        self.assertEqual(import_report(self.store, self.report)["imported"], 0)
        for row in self.report["misses"]:
            row["matched_rank"] = None
        self.store.state["annotations"]["20"] = {"reviews": {"C": {"verdict": "yes"}}}
        self.assertEqual(import_report(self.store, self.report, overwrite=True)["review_conflicts"], 1)

    def test_cli_dry_run_backup_and_reload(self):
        atomic_write(self.store.output, self.store.state)
        before = self.store.output.read_bytes()
        report = self.path / "report.json"
        atomic_write(report, self.report)
        args = ["--report", str(report), "--annotations", str(self.store.output)]
        self.assertEqual(main(args + ["--dry-run"]), 0)
        self.assertEqual(self.store.output.read_bytes(), before)
        self.assertEqual(main(args), 0)
        backups = list(self.path.glob("labels.json.*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)
        restored = Store(self.records, self.store.output, self.store.state["sources"])
        self.assertGreater(restored.summary()["genuine"], 0)


if __name__ == "__main__":
    unittest.main()
