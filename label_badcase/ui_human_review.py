"""Browser regression for human review filters, using isolated temporary labels."""
import copy
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright, expect
from server import ROOT, Store, build_dataset, handler, load


def main():
    demo = load(ROOT / "demo_data.json")
    template = build_dataset(demo["badcases"], demo["analysis"], demo["cases"])[0]
    records = []
    for i in range(1, 9):
        record = copy.deepcopy(template)
        record.update(key=str(i), sample_index=i, query=f"filter-query-{i}")
        records.append(record)
    extra = copy.deepcopy(records[2]["retrieved"][0])
    extra.update(case_id="EXTRA", rank=10, is_gt=False, model_same=False)
    records[2]["retrieved"].append(extra)
    records[7]["retrieved"] = []

    def review(verdict):
        return {"verdict": verdict, "reason": "测试判断", "owner": "测试人"}

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(records, Path(tmp) / "labels.json", {"demo": True})
        annotations = {
            "1": {"reviews": {"DEMO_RESET": review("yes")}},
            "2": {"reviews": {"DEMO_RESET": review("uncertain")}},
            "3": {"reviews": {"DEMO_RESET": review("yes"), "EXTRA": review("uncertain")}},
            "4": {"completed": True, "reviews": {"DEMO_RESET": review("no")},
                  "error_type": "其他", "error_detail": "均不适用", "owner": "测试人"},
            "5": {"reviews": {"DEMO_RESET": review("no")}},
            "6": {"reviews": {"DEMO_RESET": review("")}, "error_type": "其他"},
            "8": {"completed": True, "error_type": "其他", "error_detail": "无候选", "owner": "测试人"},
        }
        for key, annotation in annotations.items():
            store.mutate({"action": "annotation", "key": key, "record_revision": 0,
                          "annotation": annotation})
        app = ThreadingHTTPServer(("127.0.0.1", 0), handler(store))
        threading.Thread(target=app.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel="chrome", headless=True)
                try:
                    page = browser.new_page(viewport={"width": 1440, "height": 1000})
                    errors = []
                    page.on("pageerror", lambda e: errors.append(str(e)))
                    page.goto(f"http://127.0.0.1:{app.server_port}")
                    page.wait_for_load_state("networkidle")
                    human = page.locator("#human-review-filter")

                    def check(value, keys):
                        human.select_option(value)
                        expect(page.locator("#page-info")).to_contain_text(f"/ {len(keys)} 条")
                        actual = []
                        page.locator("#jump").fill("1")
                        page.locator("#jump-go").click()
                        for index, key in enumerate(keys):
                            expect(page.locator("#record-state")).to_contain_text(f"样本 {key}")
                            actual.append(key)
                            if index < len(keys) - 1:
                                page.locator("#next").click()
                        expect(page.locator("#next")).to_be_disabled()
                        assert actual == keys

                    check("yes", ["1", "3"])
                    check("uncertain", ["2", "3"])
                    check("no", ["4", "8"])
                    check("unreviewed", ["6", "7"])
                    page.locator("#error-type-filter").select_option("type:其他")
                    expect(page.locator("#page-info")).to_have_text("1 / 1 条")
                    expect(page.locator("#record-state")).to_contain_text("样本 6")
                    page.locator("#filter").select_option("complete")
                    expect(page.locator("#workspace")).to_contain_text("没有匹配的 badcase")
                    page.locator("#filter").select_option("all")
                    page.locator("#error-type-filter").select_option("")
                    human.select_option("uncertain")
                    page.locator("#search").fill("filter-query-3")
                    page.locator("#search").press("Enter")
                    expect(page.locator("#page-info")).to_have_text("1 / 1 条")
                    page.locator("#search").fill("")
                    page.locator("#search").press("Enter")
                    page.locator("#next-pending").click()
                    expect(page.locator("#record-state")).to_contain_text("样本 2")
                    # Switching filters flushes the just-edited draft before matching it.
                    page.locator('[data-cid="DEMO_RESET"] [data-verdict="yes"]').click()
                    human.select_option("yes")
                    expect(page.locator("#page-info")).to_contain_text("/ 3 条")
                    assert store.state["annotations"]["2"]["reviews"]["DEMO_RESET"]["verdict"] == "yes"
                    page.set_viewport_size({"width": 390, "height": 844})
                    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                    page.screenshot(path="/tmp/badcase-human-review-mobile.png", full_page=True)
                    assert not errors, errors
                    print("Human review UI passed: mixed verdicts, drafts, confirmed negatives, unreviewed, combined filters, paging, autosave, mobile.")
                finally:
                    browser.close()
        finally:
            app.shutdown()
            app.server_close()


if __name__ == "__main__":
    main()
